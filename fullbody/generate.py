"""Candidate generation: BRICS recombination of reference-drug fragments, filtered with RDKit, then run
through the reverse-inference loop.

HONESTY NOTES
- Generated molecules have NO measured activity. Potency/selectivity/safety checks come back as "NO DATA" flags;
  every result is labelled `hypothesis`. The only score is a similarity prior (Tanimoto to references with measured
  JAK1-over-JAK2 selectivity vs references without it). It is a ranking heuristic, not a prediction of activity.
- Synthetic accessibility (SA score) is the RDKit-contrib heuristic, not a retrosynthesis.
"""
from __future__ import annotations

import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import BRICS, QED, Crippen, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from . import ria

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_CATALOG = None
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _alerts(m) -> list[str]:
    global _CATALOG
    if _CATALOG is None:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _CATALOG = FilterCatalog(p)
    return sorted({e.GetDescription() for e in _CATALOG.GetMatches(m)})


def _fp(m):
    return _MORGAN.GetFingerprint(m)


def load_references(snapshot: dict, primary="CHEMBL2835", anti="CHEMBL2971", min_delta=1.0):
    """Split structure-bearing reference molecules by MEASURED selectivity (median pChEMBL difference)."""
    sel, non = {}, {}
    for mid, rec in snapshot["molecules"].items():
        smi = rec.get("smiles")
        p, a = rec.get("pchembl", {}).get(primary), rec.get("pchembl", {}).get(anti)
        if not (smi and p and a):
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        m = _largest_fragment(m)
        delta = statistics.median(p) - statistics.median(a)
        (sel if delta >= min_delta else non)[mid] = {"name": rec["name"], "mol": m, "delta": round(delta, 2)}
    return sel, non


def _largest_fragment(m):
    frags = Chem.GetMolFrags(m, asMols=True)
    return max(frags, key=lambda x: x.GetNumHeavyAtoms())


def brics_pool(refs: dict, max_depth=2, max_build=1000, time_cap_s=60, seed=0):
    frag_smiles = set()
    for r in refs.values():
        frag_smiles |= BRICS.BRICSDecompose(r["mol"])
    frag_mols = [Chem.MolFromSmiles(f) for f in sorted(frag_smiles)]
    frag_mols = [f for f in frag_mols if f is not None]
    t0, out = time.time(), []
    gen = BRICS.BRICSBuild(frag_mols, maxDepth=max_depth, scrambleReagents=False)
    for m in itertools.islice(gen, max_build):
        if time.time() - t0 > time_cap_s:
            break
        try:
            Chem.SanitizeMol(m)
        except Exception:  # noqa: BLE001 - invalid recombinations are expected
            continue
        out.append(m)
    return sorted(frag_smiles), out


def describe(m) -> dict:
    return {"mw": round(rdMolDescriptors.CalcExactMolWt(m), 2), "alogp": round(Crippen.MolLogP(m), 2),
            "psa": round(rdMolDescriptors.CalcTPSA(m), 2), "hbd": Lipinski.NumHDonors(m), "hba": Lipinski.NumHAcceptors(m),
            "ro5_violations": sum([rdMolDescriptors.CalcExactMolWt(m) > 500, Crippen.MolLogP(m) > 5,
                                   Lipinski.NumHDonors(m) > 5, Lipinski.NumHAcceptors(m) > 10]),
            "qed": round(QED.qed(m), 3), "sa": round(sascorer.calculateScore(m), 2), "heavy": m.GetNumHeavyAtoms()}


class GeneratedBackend:
    """Backend with computed properties and NO measurements (everything activity-related is missing)."""

    def __init__(self, snapshot: dict, mols: dict):
        self.snap, self.mols, self._desc = snapshot, mols, {k: describe(v) for k, v in mols.items()}

    def target_id(self, gene): return self.snap["targets"].get(gene)
    def candidates(self, tid): return list(self.mols)
    def name(self, mid): return mid
    def pchembl(self, mid, tid): return []
    def measurements(self, mid, tid): return {"values": [], "censored": []}

    def properties(self, mid):
        d = self._desc[mid]
        return {"mw": d["mw"], "alogp": d["alogp"], "psa": d["psa"], "hbd": d["hbd"], "hba": d["hba"],
                "ro5_violations": d["ro5_violations"], "qed": d["qed"]}


def generate(snapshot_path, profile_path, kb_path, top_n=10, max_depth=2, max_build=1000,
             sim_range=(0.30, 0.90), qed_min=0.5, sa_max=4.5, heavy_range=(15, 35)):
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    sel, non = load_references(snap)
    if not sel:
        return {"error": "no structure-bearing reference with measured selectivity", "candidates": []}
    refs = {**sel, **non}
    frags, pool = brics_pool(refs, max_depth=max_depth, max_build=max_build)
    ref_fps = {k: _fp(v["mol"]) for k, v in refs.items()}
    ref_inchi = {Chem.MolToInchiKey(v["mol"])[:14] for v in refs.values()}
    funnel = {"built": len(pool)}
    seen, kept = set(), {}
    for m in pool:
        key = Chem.MolToInchiKey(m)[:14]
        if key in seen or key in ref_inchi:
            continue
        seen.add(key)
        d = describe(m)
        if not (heavy_range[0] <= d["heavy"] <= heavy_range[1]) or d["ro5_violations"] > profile.get("property_limits", {}).get("ro5_violations_max", 1):
            continue
        if d["qed"] < qed_min or d["sa"] > sa_max or _alerts(m):
            continue
        fp = _fp(m)
        sims = {k: DataStructs.TanimotoSimilarity(fp, f) for k, f in ref_fps.items()}
        s_sel, s_non = max(sims[k] for k in sel), max((sims[k] for k in non), default=0.0)
        if not (sim_range[0] <= max(s_sel, s_non) <= sim_range[1]):
            continue
        kept[f"GEN-{len(kept) + 1:04d}"] = (m, d, round(s_sel, 3), round(s_non, 3))
    funnel.update({"unique_novel": len(seen), "passed_filters": len(kept)})
    backend = GeneratedBackend(snap, {k: v[0] for k, v in kept.items()})
    profile = {**profile, "missing_data_policy": "flag"}
    rows = []
    for gid, (m, d, ss, sn) in kept.items():
        res = ria.verify_candidate(gid, profile, backend, {g: backend.target_id(g) for g in
                                   profile["primary_targets"] + [a["gene"] for a in profile.get("anti_targets", [])]})
        rows.append({"id": gid, "smiles": Chem.MolToSmiles(m), "status": "hypothesis" if res.passed_all else "rejected",
                     "similarity_prior": round(ss - sn, 3), "sim_selective": ss, "sim_nonselective": sn,
                     "props": d, "failed": res.failed, "flags": res.flags})
    rows.sort(key=lambda r: (-(r["status"] == "hypothesis"), -r["similarity_prior"], -r["props"]["qed"]))
    return {"references": {"selective": {k: {"name": v["name"], "delta": v["delta"]} for k, v in sel.items()},
                           "non_selective": {k: {"name": v["name"], "delta": v["delta"]} for k, v in non.items()}},
            "fragments": frags, "funnel": funnel, "candidates": rows[:top_n],
            "caveat": "No measured activity for any generated molecule. 'hypothesis' means it passed property/alert filters "
                      "only; potency, selectivity and safety checks are NO DATA flags. Needs docking/assay before any claim."}
