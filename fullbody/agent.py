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
    thresholds: dict
    snapshot: dict
    mol: dict
    target: dict
    cell: dict
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


def cell_agent(state: State) -> State:
    """Occupancy at the requested free concentration + the concentration for 50% occupancy."""
    t = state["target"]
    conc = state.get("free_conc_nM")
    cell = {}
    for pw, (g, role) in PATHWAYS.items():
        ic50 = t[g]["ic50_nM"]
        cell[pw] = {"target": g, "role": role, "ic50_nM": ic50,
                    "occupancy_at_conc": None if conc is None else round(conc / (conc + ic50), 3)}
    state["cell"] = cell
    _t(state, "3_cell", "cell_agent",
       "; ".join(f"{pw}: occ={v['occupancy_at_conc']}" for pw, v in cell.items()) if conc is not None
       else "no concentration given; window analysis only", True)
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
    state["body"] = body
    _t(state, "7_body", "body_agent", f"therapeutic window (free nM): {body['window_nM']}"
       + (f"; at {conc} nM -> {body['at_conc']}" if conc is not None else ""), o["window_exists"])
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
    for name, fn in [("decompose", decompose), ("molecule", molecule_agent), ("target", target_agent),
                     ("cell", cell_agent), ("organ", organ_agent), ("body", body_agent), ("arbiter", arbiter)]:
        g.add_node(name, fn)
    g.add_edge(START, "decompose")
    g.add_conditional_edges("decompose", route, {"continue": "molecule", "arbiter": "arbiter"})
    g.add_conditional_edges("molecule", route, {"continue": "target", "arbiter": "arbiter"})
    g.add_conditional_edges("target", route, {"continue": "cell", "arbiter": "arbiter"})
    g.add_edge("cell", "organ")
    g.add_edge("organ", "body")
    g.add_edge("body", "arbiter")
    g.add_edge("arbiter", END)
    return g.compile()


def simulate(molecule_id: str, snapshot_path: str | Path, free_conc_nM: float | None = None,
             thresholds: dict | None = None) -> State:
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    st: State = {"molecule_id": molecule_id, "free_conc_nM": free_conc_nM, "snapshot": snap,
                 "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})}, "trace": []}
    return build_graph().invoke(st)
