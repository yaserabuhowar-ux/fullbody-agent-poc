"""Full-Body Agent PoC: supervisor + level agents wired as a LangGraph state machine.

Levels implemented (semi-quantitative, NOT a validated physiological simulation):
  1 molecule  - RDKit: descriptors, Ro5, structural alerts, cross-check vs ChEMBL values
  3 cell      - target occupancy from ChEMBL IC50 (occ = C / (C + IC50)); pathway inhibition
  5 organ     - synovium efficacy proxy (JAK1) vs bone-marrow liability proxy (JAK2)
  7 body      - therapeutic-window test; safety-first arbitration with trace chain
Levels 2, 4, 6 and the hormonal/mineral/neural modulation layers are NOT modelled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from . import pk
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

TARGETS = {"JAK1": "CHEMBL2835", "JAK2": "CHEMBL2971"}
# pathway -> (target, role). Curated seed knowledge, not inferred.
PATHWAYS = {
    "IL-6/STAT3 (inflammation)": ("JAK1", "efficacy"),
    "EPO/STAT5 (erythropoiesis)": ("JAK2", "liability"),
}
DEFAULT_THRESHOLDS = {"occ_efficacy_min": 0.5, "occ_liability_max": 0.2, "ro5_max": 1, "qed_min": 0.5}


class State(TypedDict, total=False):
    molecule_id: str
    free_conc_nM: float | None
    use_pk: bool
    pk_path: str
    pk: dict
    thresholds: dict
    snapshot: dict
    mol: dict
    target: dict
    cell: dict
    cell_sensitivity: dict
    organ: dict
    body: dict
    verdict: dict
    trace: list
    halt: bool


def _t(state, level, agent, msg, ok=None):
    state.setdefault("trace", []).append({"level": level, "agent": agent, "finding": msg, "ok": ok})


# ---------------------------------------------------------------- level agents
def decompose(state: State) -> State:
    rec = state["snapshot"]["molecules"].get(state["molecule_id"])
    if rec is None:
        state["halt"] = True
        _t(state, "supervisor", "decompose", f"{state['molecule_id']} not in snapshot", False)
    else:
        _t(state, "supervisor", "decompose", f"simulate {rec['name']} across levels 1,3,5,7", True)
    return state


def molecule_agent(state: State) -> State:
    rec = state["snapshot"]["molecules"][state["molecule_id"]]
    smi = rec.get("smiles")
    if not smi:
        state["halt"] = True
        state["mol"] = {"error": "no structure"}
        _t(state, "1_molecule", "molecule_agent", "no SMILES in snapshot; cannot run RDKit layer", False)
        return state
    m = Chem.MolFromSmiles(smi)
    if m is None:
        state["halt"] = True
        state["mol"] = {"error": "invalid SMILES"}
        _t(state, "1_molecule", "molecule_agent", "SMILES failed to parse", False)
        return state
    desc = {"mw": round(Descriptors.MolWt(m), 2), "logp": round(Crippen.MolLogP(m), 2),
            "tpsa": round(rdMolDescriptors.CalcTPSA(m), 2), "hbd": Lipinski.NumHDonors(m),
            "hba": Lipinski.NumHAcceptors(m), "rotb": Lipinski.NumRotatableBonds(m)}
    ro5 = sum([desc["mw"] > 500, desc["logp"] > 5, desc["hbd"] > 5, desc["hba"] > 10])
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    alerts = sorted({e.GetDescription() for e in FilterCatalog(params).GetMatches(m)})
    ref = rec.get("properties", {})
    xcheck = {"mw_diff": round(desc["mw"] - ref["mw"], 2)} if "mw" in ref else {}
    qed = ref.get("qed")
    state["mol"] = {"descriptors": desc, "ro5_violations": ro5, "alerts": alerts, "qed_chembl": qed, "xcheck": xcheck}
    th = state["thresholds"]
    ok = ro5 <= th["ro5_max"] and (qed is None or qed >= th["qed_min"])
    _t(state, "1_molecule", "molecule_agent",
       f"MW {desc['mw']}, cLogP {desc['logp']}, TPSA {desc['tpsa']}, Ro5 violations {ro5}, QED(ChEMBL) {qed}, "
       f"structural alerts: {alerts or 'none'}", ok)
    return state


def target_agent(state: State) -> State:
    rec = state["snapshot"]["molecules"][state["molecule_id"]]
    out = {}
    for g, tid in TARGETS.items():
        vals = sorted(rec.get("pchembl", {}).get(tid, []))
        if not vals:
            out[g] = None
            continue
        med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
        out[g] = {"pchembl_median": round(med, 2), "ic50_nM": round(10 ** (9 - med), 1),
                  "n": len(vals), "spread_pchembl": [vals[0], vals[-1]]}
    state["target"] = out
    missing = [g for g, v in out.items() if v is None]
    if missing:
        state["halt"] = True
    _t(state, "1_molecule", "target_agent",
       "; ".join(f"{g}: IC50~{v['ic50_nM']} nM (n={v['n']}, pChEMBL {v['spread_pchembl'][0]}-{v['spread_pchembl'][1]})"
                 for g, v in out.items() if v) + (f"; missing {missing}" if missing else ""), not missing)
    return state


def pk_agent(state: State) -> State:
    """Level 'whole-body PK': dose -> steady-state free concentration profile (one-compartment)."""
    if not state.get("use_pk"):
        return state
    params = json.loads(Path(state["pk_path"]).read_text(encoding="utf-8")).get(state["molecule_id"])
    if params is None:
        state["halt"] = True
        _t(state, "7_body", "pk_agent", "no sourced PK parameters for this molecule", False)
        return state
    m = pk.model_from_params(params)
    d = params["regimen"]
    ts, prof = pk.steady_state_profile(d["dose_mg"], d["interval_h"], m)
    summ = pk.summarize(ts, prof)
    fu, mw = params["fu"]["value"], params["mw"]
    free = [pk.ng_ml_to_free_nM(c, mw, fu) for c in prof]
    state["pk"] = {"regimen": d, "model": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                   "summary": {k: round(v, 2) for k, v in summ.items()}, "free_nM": free, "t_h": ts,
                   "free_cmax_nM": round(max(free), 1), "free_cmin_nM": round(min(free), 1),
                   "free_mean_nM": round(sum(free) / len(free), 1), "params": params}
    _t(state, "7_body", "pk_agent",
       f"{d['dose_mg']} mg q{d['interval_h']}h: total Cmax {summ['cmax_ng_ml']:.1f} ng/mL, "
       f"free Cmax {max(free):.1f} nM / mean {sum(free)/len(free):.1f} nM / trough {min(free):.1f} nM "
       f"(t1/2 used {m['t_half_used_h']:.2f} h{', fitted to label Cmax' if m['fitted_to_label_cmax'] else ''})", True)
    return state


def _mean_occ(free, ic50):
    return sum(c / (c + ic50) for c in free) / len(free)


def cell_agent(state: State) -> State:
    """Occupancy at the requested free concentration + the concentration for 50% occupancy."""
    t = state["target"]
    conc = state.get("free_conc_nM")
    cell = {}
    for pw, (g, role) in PATHWAYS.items():
        ic50 = t[g]["ic50_nM"]
        cell[pw] = {"target": g, "role": role, "ic50_nM": ic50,
                    "occupancy_at_conc": None if conc is None else round(conc / (conc + ic50), 3)}
    if state.get("pk"):
        free = state["pk"]["free_nM"]
        for pw, v in cell.items():
            ic = v["ic50_nM"]
            v["occupancy_mean"] = round(_mean_occ(free, ic), 3)
            v["occupancy_peak"] = round(max(free) / (max(free) + ic), 3)
            v["occupancy_trough"] = round(min(free) / (min(free) + ic), 3)
        # sensitivity over the IC50 spread of the ChEMBL records (best/worst case)
        sens = {}
        for g in TARGETS:
            lo_p, hi_p = t[g]["spread_pchembl"]
            sens[g] = {"ic50_range_nM": [round(10 ** (9 - hi_p), 1), round(10 ** (9 - lo_p), 1)]}
        th = state["thresholds"]
        cases = {}
        for label, j1, j2 in [("favourable", sens["JAK1"]["ic50_range_nM"][0], sens["JAK2"]["ic50_range_nM"][1]),
                              ("unfavourable", sens["JAK1"]["ic50_range_nM"][1], sens["JAK2"]["ic50_range_nM"][0])]:
            cases[label] = {"jak1_mean_occ": round(_mean_occ(free, j1), 3), "jak2_mean_occ": round(_mean_occ(free, j2), 3)}
            cases[label]["passes"] = (cases[label]["jak1_mean_occ"] >= th["occ_efficacy_min"]
                                      and cases[label]["jak2_mean_occ"] <= th["occ_liability_max"])
        state["cell_sensitivity"] = cases
    state["cell"] = cell
    if state.get("pk"):
        msg = "; ".join(f"{pw}: mean occ={v['occupancy_mean']} (peak {v['occupancy_peak']}, trough {v['occupancy_trough']})"
                        for pw, v in cell.items())
    elif conc is not None:
        msg = "; ".join(f"{pw}: occ={v['occupancy_at_conc']}" for pw, v in cell.items())
    else:
        msg = "no concentration given; window analysis only"
    _t(state, "3_cell", "cell_agent", msg, True)
    return state


def organ_agent(state: State) -> State:
    th, t = state["thresholds"], state["target"]
    j1, j2 = t["JAK1"]["ic50_nM"], t["JAK2"]["ic50_nM"]
    # concentration where JAK1 occ = occ_efficacy_min, and where JAK2 occ = occ_liability_max
    c_eff = j1 * th["occ_efficacy_min"] / (1 - th["occ_efficacy_min"])
    c_safe = j2 * th["occ_liability_max"] / (1 - th["occ_liability_max"])
    state["organ"] = {"c_efficacy_nM": round(c_eff, 1), "c_liability_limit_nM": round(c_safe, 1),
                      "window_exists": c_eff <= c_safe, "fold_selectivity": round(j2 / j1, 2)}
    o = state["organ"]
    _t(state, "5_organ", "organ_agent",
       f"synovium needs >= {o['c_efficacy_nM']} nM free (JAK1 occ {th['occ_efficacy_min']}); "
       f"marrow liability limit <= {o['c_liability_limit_nM']} nM (JAK2 occ {th['occ_liability_max']}); "
       f"JAK2/JAK1 IC50 ratio {o['fold_selectivity']}x", o["window_exists"])
    return state


def body_agent(state: State) -> State:
    o, conc, th = state["organ"], state.get("free_conc_nM"), state["thresholds"]
    body = {"window_nM": [o["c_efficacy_nM"], o["c_liability_limit_nM"]] if o["window_exists"] else None}
    if conc is not None:
        cell = state["cell"]
        eff = cell["IL-6/STAT3 (inflammation)"]["occupancy_at_conc"]
        liab = cell["EPO/STAT5 (erythropoiesis)"]["occupancy_at_conc"]
        body["at_conc"] = {"conc_nM": conc, "efficacy_ok": eff >= th["occ_efficacy_min"],
                           "liability_ok": liab <= th["occ_liability_max"]}
    ok = o["window_exists"]
    extra = ""
    if state.get("pk"):
        cell = state["cell"]
        eff = cell["IL-6/STAT3 (inflammation)"]["occupancy_mean"]
        liab = cell["EPO/STAT5 (erythropoiesis)"]["occupancy_mean"]
        sens = state["cell_sensitivity"]
        body["at_regimen"] = {"efficacy_mean_occ": eff, "liability_mean_occ": liab,
                              "efficacy_ok": eff >= th["occ_efficacy_min"], "liability_ok": liab <= th["occ_liability_max"],
                              "robust_across_ic50_spread": {k: v["passes"] for k, v in sens.items()}}
        ok = ok and body["at_regimen"]["efficacy_ok"] and body["at_regimen"]["liability_ok"]
        extra = f"; regimen: JAK1 mean occ {eff}, JAK2 mean occ {liab}; sensitivity {body['at_regimen']['robust_across_ic50_spread']}"
    state["body"] = body
    _t(state, "7_body", "body_agent", f"therapeutic window (free nM): {body['window_nM']}"
       + (f"; at {conc} nM -> {body['at_conc']}" if conc is not None else "") + extra, ok)
    return state


def arbiter(state: State) -> State:
    """Safety-first: any failed level blocks a positive verdict; reason is logged."""
    failed = [t for t in state.get("trace", []) if t["ok"] is False]
    if state.get("halt"):
        v = {"decision": "INCONCLUSIVE", "reason": "pipeline halted: missing structure/target data"}
    elif failed:
        v = {"decision": "NOT SUPPORTED", "reason": [f"{t['level']}: {t['agent']}" for t in failed]}
    else:
        v = {"decision": "SUPPORTED (computational screen only)", "reason": "all modelled levels passed"}
    v["next_experiment"] = ("whole-blood pSTAT3 (IL-6) vs pSTAT5 (EPO) dose-response with measured free "
                            "concentration; confirm JAK1/JAK2 IC50 in a single matched-ATP enzyme assay")
    v["caveat"] = ("IC50 from mixed ChEMBL assays; occupancy = C/(C+IC50) ignores ATP competition, protein "
                   "binding, PK and tissue distribution. Levels 2,4,6 and modulation layers not modelled.")
    state["verdict"] = v
    _t(state, "supervisor", "arbiter", v["decision"], failed == [] and not state.get("halt"))
    return state


def route(state: State) -> str:
    return "arbiter" if state.get("halt") else "continue"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("decompose", decompose), ("molecule", molecule_agent), ("target", target_agent), ("pk", pk_agent),
                     ("cell", cell_agent), ("organ", organ_agent), ("body", body_agent), ("arbiter", arbiter)]:
        g.add_node(name, fn)
    g.add_edge(START, "decompose")
    g.add_conditional_edges("decompose", route, {"continue": "molecule", "arbiter": "arbiter"})
    g.add_conditional_edges("molecule", route, {"continue": "target", "arbiter": "arbiter"})
    g.add_conditional_edges("target", route, {"continue": "pk", "arbiter": "arbiter"})
    g.add_conditional_edges("pk", route, {"continue": "cell", "arbiter": "arbiter"})
    g.add_edge("cell", "organ")
    g.add_edge("organ", "body")
    g.add_edge("body", "arbiter")
    g.add_edge("arbiter", END)
    return g.compile()


def simulate(molecule_id: str, snapshot_path: str | Path, free_conc_nM: float | None = None,
             thresholds: dict | None = None, use_pk: bool = False, pk_path: str | Path | None = None) -> State:
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    st: State = {"molecule_id": molecule_id, "free_conc_nM": free_conc_nM, "snapshot": snap,
                 "use_pk": use_pk, "pk_path": str(pk_path or Path(snapshot_path).with_name("pk_params.json")),
                 "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})}, "trace": []}
    return build_graph().invoke(st)
