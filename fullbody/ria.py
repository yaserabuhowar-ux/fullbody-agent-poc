"""Reverse Inference Agent (RIA) - prototype.

Takes a "desired profile" (target engagement + selectivity + drug-likeness limits),
traces it from body level (7) down to molecule level (1) using a curated pathway
knowledge file, pulls candidate molecules from a ChEMBL backend, verifies them
upward against the profile, and arbitrates with a safety-first rule.

Scope / honesty notes
- This RANKS KNOWN ChEMBL compounds against a profile. It does not generate new molecules.
- The level chain (7->1) is read from knowledge/pathways.json (human-curated seed), not inferred.
- Selectivity here = median pChEMBL(primary) - median pChEMBL(anti-target) over the
  records the backend returns. Mixed assay formats make this a screening signal only.
- RestBackend is written against the public ChEMBL REST API but was NOT run live in the
  build sandbox (network blocked); SnapshotBackend is what the tests exercise.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

LEVEL_ORDER = ["7_body", "6_organ_system", "5_organ", "4_tissue", "3_cell", "2_organelle", "1_molecule"]


# --------------------------------------------------------------------------- backends
class ChemblBackend(Protocol):
    def target_id(self, gene: str) -> str | None: ...
    def candidates(self, target_chembl_id: str) -> list[str]: ...
    def pchembl(self, molecule_id: str, target_chembl_id: str) -> list[float]: ...
    def properties(self, molecule_id: str) -> dict: ...
    def name(self, molecule_id: str) -> str: ...
    def measurements(self, molecule_id: str, target_chembl_id: str) -> dict: ...


class SnapshotBackend:
    """Reads a frozen JSON snapshot (see data/jak_snapshot.json)."""

    def __init__(self, path: str | Path):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))

    def target_id(self, gene):
        return self.data["targets"].get(gene)

    def candidates(self, target_chembl_id):
        return [m for m, v in self.data["molecules"].items() if target_chembl_id in v.get("pchembl", {})]

    def pchembl(self, molecule_id, target_chembl_id):
        return list(self.data["molecules"][molecule_id].get("pchembl", {}).get(target_chembl_id, []))

    def properties(self, molecule_id):
        return dict(self.data["molecules"][molecule_id].get("properties", {}))

    def name(self, molecule_id):
        return self.data["molecules"][molecule_id].get("name", molecule_id)

    def measurements(self, molecule_id, target_chembl_id):
        m = self.data["molecules"][molecule_id]
        return {"values": list(m.get("pchembl", {}).get(target_chembl_id, [])),
                "censored": [list(x) for x in m.get("censored", {}).get(target_chembl_id, [])]}


class RestBackend:
    """Live ChEMBL REST backend (https://www.ebi.ac.uk/chembl/api/data).

    NOT exercised against the real service in the build sandbox (network blocked); covered by a mocked-session test only.
    Features: pagination, retry with backoff, optional on-disk cache, generic User-Agent.
    """

    BASE = "https://www.ebi.ac.uk/chembl/api/data"
    KNOWN_TARGETS = {"JAK1": "CHEMBL2835", "JAK2": "CHEMBL2971", "JAK3": "CHEMBL2148", "TYK2": "CHEMBL3553",
                     "hERG": "CHEMBL240", "KCNH2": "CHEMBL240", "CYP3A4": "CHEMBL340"}

    def __init__(self, session=None, min_pchembl_pool: float = 6.0, pool_limit: int = 200,
                 cache_dir: str | Path | None = None, retries: int = 3, backoff_s: float = 1.5, sleep=None):
        import time

        import requests

        self.s = session or requests.Session()
        if hasattr(self.s, "headers"):
            self.s.headers.update({"User-Agent": "fullbody-agent-poc/0.3"})
        self.min_pool, self.limit = min_pchembl_pool, pool_limit
        self.cache = Path(cache_dir) if cache_dir else None
        self.retries, self.backoff = retries, backoff_s
        self._sleep = sleep or time.sleep
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _get(self, path, **params):
        import hashlib

        key = None
        if self.cache:
            key = self.cache / (hashlib.sha1(json.dumps([path, params], sort_keys=True).encode()).hexdigest() + ".json")
            if key.exists():
                return json.loads(key.read_text(encoding="utf-8"))
        last = None
        for attempt in range(self.retries):
            try:
                r = self.s.get(f"{self.BASE}/{path}.json", params=params, timeout=60)
                if getattr(r, "status_code", 200) in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                js = r.json()
                if key:
                    key.write_text(json.dumps(js), encoding="utf-8")
                return js
            except Exception as e:  # noqa: BLE001 - retry any transport/HTTP error, re-raise after the last try
                last = e
                self._sleep(self.backoff * (2 ** attempt))
        raise last

    def _pages(self, path, key, **params):
        out, offset = [], 0
        while True:
            js = self._get(path, limit=100, offset=offset, **params)
            out.extend(js.get(key, []))
            if not (js.get("page_meta") or {}).get("next"):
                return out
            offset += 100

    def target_id(self, gene):
        if gene in self.KNOWN_TARGETS:
            return self.KNOWN_TARGETS[gene]
        js = self._get("target", target_components__target_component_synonyms__component_synonym__iexact=gene,
                       target_type="SINGLE PROTEIN", organism="Homo sapiens", limit=1)
        t = js.get("targets", [])
        return t[0]["target_chembl_id"] if t else None

    def candidates(self, target_chembl_id):
        acts = self._pages("activity", "activities", target_chembl_id=target_chembl_id, standard_type="IC50",
                           pchembl_value__gte=self.min_pool)[: self.limit]
        return sorted({a["molecule_chembl_id"] for a in acts})

    def measurements(self, molecule_id, target_chembl_id):
        acts = self._pages("activity", "activities", molecule_chembl_id=molecule_id, target_chembl_id=target_chembl_id,
                           standard_type="IC50")
        vals = [float(a["pchembl_value"]) for a in acts if a.get("pchembl_value") and a.get("standard_relation") in ("=", None)]
        cens = [[a["standard_relation"], float(a["standard_value"])] for a in acts
                if a.get("standard_relation") in (">", ">=") and a.get("standard_value") is not None
                and a.get("standard_units") == "nM"]
        return {"values": vals, "censored": cens}

    def pchembl(self, molecule_id, target_chembl_id):
        return self.measurements(molecule_id, target_chembl_id)["values"]

    def properties(self, molecule_id):
        m = self._get(f"molecule/{molecule_id}")
        p = m.get("molecule_properties") or {}

        def f(k):
            return float(p[k]) if p.get(k) is not None else None

        return {"mw": f("mw_freebase"), "alogp": f("alogp"), "psa": f("psa"), "hbd": f("hbd"),
                "hba": f("hba"), "ro5_violations": f("num_ro5_violations"), "qed": f("qed_weighted")}

    def name(self, molecule_id):
        return self._get(f"molecule/{molecule_id}").get("pref_name") or molecule_id


# --------------------------------------------------------------------------- data classes
@dataclass
class Check:
    level: str
    name: str
    passed: bool
    value: object
    threshold: object
    note: str = ""


@dataclass
class Result:
    chembl_id: str
    name: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def passed_all(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


# --------------------------------------------------------------------------- agent steps
def check_consistency(profile: dict) -> dict:
    """Reject profiles that contradict themselves BEFORE any search."""
    errors, warnings = [], []
    primary = set(profile.get("primary_targets", []))
    anti = {a["gene"]: a for a in profile.get("anti_targets", [])}
    if not primary:
        errors.append("no primary_targets given")
    for g in primary & set(anti):
        errors.append(f"{g} is both a primary target and an anti-target")
    for g, a in anti.items():
        if a.get("min_selectivity_delta", 0) < 0:
            errors.append(f"anti-target {g}: min_selectivity_delta must be >= 0")
        kinds = [k for k in ("min_selectivity_delta", "min_ic50_uM", "informational") if k in a]
        if len(kinds) != 1:
            errors.append(f"anti-target {g}: give exactly one of min_selectivity_delta / min_ic50_uM / informational")
        if a.get("min_ic50_uM", 1) <= 0:
            errors.append(f"anti-target {g}: min_ic50_uM must be > 0")
    if profile.get("missing_data_policy", "fail") not in ("fail", "pass", "flag"):
        errors.append("missing_data_policy must be fail, pass or flag")
    if profile.get("min_pchembl") is None:
        warnings.append("min_pchembl missing; potency will not be enforced")
    levels = profile.get("levels", {})
    if "7_body" not in levels or "1_molecule" not in levels:
        warnings.append("profile should state at least level 7 (body) and level 1 (molecule)")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def reverse_trace(profile: dict, kb_path: str | Path) -> dict:
    """Read the curated 7->1 chain for the indication; flag disagreement with the profile."""
    kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    entry = kb.get(profile.get("indication", "").lower())
    if not entry:
        return {"found": False, "chain": {}, "note": "indication not in knowledge file; chain not available"}
    note = ""
    if set(entry["primary_targets"]) != set(profile.get("primary_targets", [])):
        note = (f"profile targets {profile.get('primary_targets')} differ from "
                f"knowledge-file targets {entry['primary_targets']}; profile wins")
    return {"found": True, "chain": entry["chain"], "kb_anti_targets": entry.get("anti_targets", []),
            "rationale": entry.get("anti_target_rationale", ""), "note": note}


def _median(xs):
    return statistics.median(xs) if xs else None


def verify_candidate(mol: str, profile: dict, backend: ChemblBackend, tids: dict) -> Result:
    res = Result(mol, backend.name(mol))
    policy = profile.get("missing_data_policy", "fail")
    ok_missing = policy == "pass"
    def _flag_missing(name, thr, why, level="3_cell"):
        """Apply the missing-data policy; returns the Check."""
        if policy == "flag":
            res.flags.append(f"{name}: NO DATA ({why})")
        return Check(level, name, policy in ("pass", "flag"), None, thr, f"no data (policy: {policy})")


    # level 1 (molecule): potency on every primary target
    pots = []
    for g in profile["primary_targets"]:
        vals = backend.pchembl(mol, tids[g])
        med = _median(vals)
        pots.append(med)
        res.metrics[f"pchembl_{g}"] = med
        res.metrics[f"n_{g}"] = len(vals)
        thr = profile.get("min_pchembl")
        if med is None:
            res.checks.append(_flag_missing(f"potency_{g}", thr, "no measurement", level="1_molecule"))
        else:
            res.checks.append(Check("1_molecule", f"potency_{g}", thr is None or med >= thr, med, thr))
    primary_med = min(p for p in pots if p is not None) if all(p is not None for p in pots) else None

    # levels 3-5: anti-targets (selectivity, safety floor, or informational)
    for a in profile.get("anti_targets", []):
        g = a["gene"]
        meas = backend.measurements(mol, tids[g])
        vals, cens = meas["values"], meas["censored"]
        med = _median(vals)
        res.metrics[f"pchembl_{g}"] = med
        res.metrics[f"n_{g}"] = len(vals)
        if "min_selectivity_delta" in a:
            if med is None or primary_med is None:
                res.checks.append(_flag_missing(f"selectivity_vs_{g}", a["min_selectivity_delta"], "no pChEMBL"))
                continue
            delta = round(primary_med - med, 2)
            res.metrics[f"delta_vs_{g}"] = delta
            res.checks.append(Check("3_cell", f"selectivity_vs_{g}", delta >= a["min_selectivity_delta"],
                                    delta, a["min_selectivity_delta"]))
        elif "min_ic50_uM" in a:
            floor = a["min_ic50_uM"]
            if vals:
                ic50_uM = round(10 ** (9 - med) / 1000.0, 3)
                res.metrics[f"ic50_uM_{g}"] = ic50_uM
                res.checks.append(Check("3_cell", f"safety_floor_{g}", ic50_uM >= floor, ic50_uM, floor, "measured IC50"))
            elif cens and max(c[1] for c in cens if c[0] in (">", ">=")) / 1000.0 >= floor:
                bound = max(c[1] for c in cens if c[0] in (">", ">=")) / 1000.0
                res.metrics[f"ic50_uM_{g}"] = f">{bound}"
                res.checks.append(Check("3_cell", f"safety_floor_{g}", True, f">{bound}", floor, "censored bound meets floor"))
            else:
                why = "only censored bounds below the floor" if cens else "no records"
                res.checks.append(_flag_missing(f"safety_floor_{g}", floor, why))
        else:  # informational: recorded, never gates
            res.metrics[f"info_{g}_pchembl"] = med
            if med is None:
                res.flags.append(f"{g}: informational, no data")

    # levels 5-7: drug-likeness limits
    props = backend.properties(mol)
    res.metrics.update({f"prop_{k}": v for k, v in props.items()})
    lim = profile.get("property_limits", {})
    rules = [("ro5_violations", "ro5_violations_max", lambda v, t: v <= t),
             ("qed", "qed_min", lambda v, t: v >= t),
             ("psa", "psa_max", lambda v, t: v <= t)]
    for key, lk, fn in rules:
        if lk in lim:
            v = props.get(key)
            passed = ok_missing if v is None else fn(v, lim[lk])
            res.checks.append(Check("5_organ", f"property_{key}", passed, v, lim[lk]))
    return res


def arbitrate(results: list[Result], profile: dict) -> list[dict]:
    """Safety-first: hard-fail on any failed check; survivors ranked by selectivity, potency, QED."""
    anti = [a["gene"] for a in profile.get("anti_targets", []) if "min_selectivity_delta" in a]
    prim = profile["primary_targets"]

    def key(r: Result):
        sel = min((r.metrics.get(f"delta_vs_{g}", -99) for g in anti), default=0)
        pot = min((r.metrics.get(f"pchembl_{g}") or 0 for g in prim))
        return (-sel, -pot, -(r.metrics.get("prop_qed") or 0))

    survivors = sorted([r for r in results if r.passed_all], key=key)
    rejected = [r for r in results if not r.passed_all]
    out = []
    for i, r in enumerate(survivors, 1):
        out.append({"rank": i, "status": "shortlisted", **_dump(r)})
    for r in sorted(rejected, key=lambda r: r.name):
        out.append({"rank": None, "status": "rejected", "reason": r.failed, **_dump(r)})
    return out


def _dump(r: Result) -> dict:
    return {"chembl_id": r.chembl_id, "name": r.name, "metrics": r.metrics, "flags": r.flags,
            "trace": [asdict(c) for c in r.checks]}


def run(profile: dict, backend: ChemblBackend, kb_path: str | Path) -> dict:
    cons = check_consistency(profile)
    if not cons["ok"]:
        return {"stage": "consistency", "consistency": cons, "ranking": []}
    trace = reverse_trace(profile, kb_path)
    genes = list(profile["primary_targets"]) + [a["gene"] for a in profile.get("anti_targets", [])]
    tids = {g: backend.target_id(g) for g in genes}
    missing = [g for g, t in tids.items() if t is None]
    if missing:
        return {"stage": "target_resolution", "consistency": cons, "error": f"unresolved genes: {missing}",
                "ranking": []}
    pool: list[str] = []
    for g in profile["primary_targets"]:
        for m in backend.candidates(tids[g]):
            if m not in pool:
                pool.append(m)
    results = [verify_candidate(m, profile, backend, tids) for m in pool]
    return {"stage": "done", "consistency": cons, "reverse_trace": trace, "targets": tids,
            "n_candidates": len(pool), "ranking": arbitrate(results, profile)}


# --------------------------------------------------------------------------- report
def to_markdown(out: dict, profile: dict) -> str:
    L = [f"# تقرير وكيل الاستدلال العكسي — {profile.get('indication')}", ""]
    if out["stage"] != "done":
        L += [f"توقف عند المرحلة: **{out['stage']}**", "", json.dumps(out, ensure_ascii=False, indent=2)]
        return "\n".join(L)
    L += ["## السلسلة العكسية (7 ← 1) — معرفة مُنسَّقة يدويًا، لم يستنتجها الوكيل", ""]
    for k in LEVEL_ORDER:
        if k in out["reverse_trace"].get("chain", {}):
            L.append(f"- **{k}**: {out['reverse_trace']['chain'][k]}")
    if out["reverse_trace"].get("note"):
        L.append(f"\n> تنبيه: {out['reverse_trace']['note']}")
    L += ["", "## الترتيب النهائي", "",
          "| الترتيب | المركّب | pChEMBL (الهدف) | الفرق عن الهدف المضاد | QED | الحالة |", "|---|---|---|---|---|---|"]
    prim = profile["primary_targets"][0]
    anti = next((a["gene"] for a in profile.get("anti_targets", []) if "min_selectivity_delta" in a), None)
    for r in out["ranking"]:
        m = r["metrics"]
        L.append(f"| {r['rank'] or '—'} | {r['name']} ({r['chembl_id']}) | {m.get('pchembl_' + prim)} | "
                 f"{m.get('delta_vs_' + anti) if anti else '—'} | {m.get('prop_qed')} | "
                 f"{'مُرشَّح' if r['status'] == 'shortlisted' else 'مرفوض: ' + ', '.join(r['reason'])} |")
    L += ["", "## سلسلة التتبع لكل مركّب", ""]
    for r in out["ranking"]:
        L.append(f"### {r['name']}")
        for c in r["trace"]:
            L.append(f"- [{'✓' if c['passed'] else '✗'}] {c['name']}: القيمة={c['value']} / الحد={c['threshold']} {c['note']}")
        L.append("")
    L += ["## حدود النتيجة", "",
          "- الوكيل يرتّب مركّبات معروفة في ChEMBL؛ لا يصمّم جزيئات جديدة.",
          "- الانتقائية = فرق وسيط pChEMBL على السجلات المُرجَعة، بأنواع فحوص مختلطة؛ مؤشر فرز أوّلي لا دليل سريري.",
          "- يلزم تحقق تجريبي ومراجعة خبير قبل أي استخدام يتجاوز العرض التوضيحي."]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--backend", choices=["snapshot", "rest"], default="snapshot")
    ap.add_argument("--snapshot", default="data/jak_snapshot.json")
    ap.add_argument("--kb", default="knowledge/pathways.json")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    profile = json.loads(Path(a.profile).read_text(encoding="utf-8"))
    backend = SnapshotBackend(a.snapshot) if a.backend == "snapshot" else RestBackend()
    out = run(profile, backend, a.kb)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (od / "report.md").write_text(to_markdown(out, profile), encoding="utf-8")
    print(f"stage={out['stage']} candidates={out.get('n_candidates')} -> {od}/result.json, report.md")
    for r in out.get("ranking", []):
        print(f"  {r['rank'] or '-':>2}  {r['name']:<14} {r['status']:<11} {r.get('reason', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
