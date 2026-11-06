import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullbody import ria  # noqa: E402

PROFILE = json.loads((ROOT / "profiles/ra_jak1.json").read_text(encoding="utf-8"))
BACKEND = ria.SnapshotBackend(ROOT / "data/jak_snapshot.json")
KB = ROOT / "knowledge/pathways.json"


def test_consistency_rejects_target_that_is_also_anti_target():
    p = copy.deepcopy(PROFILE)
    p["anti_targets"] = [{"gene": "JAK1", "min_selectivity_delta": 1.0}]
    out = ria.run(p, BACKEND, KB)
    assert out["stage"] == "consistency"
    assert any("both a primary target and an anti-target" in e for e in out["consistency"]["errors"])


def test_consistency_rejects_negative_delta():
    p = copy.deepcopy(PROFILE)
    p["anti_targets"][0]["min_selectivity_delta"] = -0.5
    assert not ria.check_consistency(p)["ok"]


def test_ranking_expected_shortlist_and_rejections():
    out = ria.run(PROFILE, BACKEND, KB)
    assert out["stage"] == "done"
    ranked = [r["name"] for r in out["ranking"] if r["status"] == "shortlisted"]
    rejected = {r["name"] for r in out["ranking"] if r["status"] == "rejected"}
    assert ranked == ["filgotinib", "abrocitinib", "upadacitinib"]
    assert rejected == {"tofacitinib", "baricitinib"}


def test_hand_computed_selectivity_values():
    out = ria.run(PROFILE, BACKEND, KB)
    m = {r["name"]: r["metrics"] for r in out["ranking"]}
    assert m["filgotinib"]["delta_vs_JAK2"] == 1.63    # 7.25 - 5.62
    assert m["abrocitinib"]["delta_vs_JAK2"] == 1.45   # 7.54 - 6.09
    assert m["upadacitinib"]["delta_vs_JAK2"] == 1.15  # 7.37 - 6.22
    assert m["tofacitinib"]["delta_vs_JAK2"] == -0.74  # 6.96 - 7.70


def test_tofacitinib_fails_potency_and_selectivity():
    out = ria.run(PROFILE, BACKEND, KB)
    toff = next(r for r in out["ranking"] if r["name"] == "tofacitinib")
    assert set(toff["reason"]) == {"potency_JAK1", "selectivity_vs_JAK2"}


def test_missing_antitarget_data_fails_by_default_and_passes_when_allowed():
    p = copy.deepcopy(PROFILE)
    p["anti_targets"] = [{"gene": "JAK2", "min_selectivity_delta": 1.0}]

    class NoJak2(ria.SnapshotBackend):
        def measurements(self, mol, tid):
            if tid == self.data["targets"]["JAK2"]:
                return {"values": [], "censored": []}
            return super().measurements(mol, tid)

    b = NoJak2(ROOT / "data/jak_snapshot.json")
    assert not any(r["status"] == "shortlisted" for r in ria.run(p, b, KB)["ranking"])
    p["missing_data_policy"] = "pass"
    assert any(r["status"] == "shortlisted" for r in ria.run(p, b, KB)["ranking"])


def test_unresolved_gene_stops_cleanly():
    p = copy.deepcopy(PROFILE)
    p["primary_targets"] = ["NOSUCHGENE"]
    out = ria.run(p, BACKEND, KB)
    assert out["stage"] == "target_resolution"


def test_rest_backend_parsing_with_mocked_session():
    class R:
        def __init__(self, js): self.js = js
        def raise_for_status(self): pass
        def json(self): return self.js

    class S:
        def get(self, url, params=None, timeout=None):
            if url.endswith("/activity.json"):
                return R({"activities": [{"molecule_chembl_id": "CHEMBL1", "pchembl_value": "7.5"},
                                         {"molecule_chembl_id": "CHEMBL1", "pchembl_value": None}]})
            if "/molecule/" in url:
                return R({"pref_name": "X", "molecule_properties": {"mw_freebase": "300", "qed_weighted": "0.8",
                          "num_ro5_violations": 0, "psa": "80", "alogp": "2", "hbd": 1, "hba": 4}})
            return R({"targets": [{"target_chembl_id": "CHEMBL9"}]})

    b = ria.RestBackend(session=S())
    assert b.target_id("JAK1") == "CHEMBL2835"   # known-ID shortcut
    assert b.target_id("FOO") == "CHEMBL9"        # falls back to the REST query
    assert b.candidates("CHEMBL9") == ["CHEMBL1"]
    assert b.pchembl("CHEMBL1", "CHEMBL9") == [7.5]
    assert b.properties("CHEMBL1")["qed"] == 0.8
    assert b.name("CHEMBL1") == "X"


EXT = json.loads((ROOT / "profiles/ra_jak1_safety.json").read_text(encoding="utf-8"))


def _by_name(out):
    return {r["name"]: r for r in out["ranking"]}


def test_extended_profile_shortlist_and_flags():
    out = ria.run(EXT, BACKEND, KB)
    r = _by_name(out)
    assert {n for n, x in r.items() if x["status"] == "shortlisted"} == {"abrocitinib", "upadacitinib", "filgotinib"}
    assert r["filgotinib"]["flags"] == []                       # has hERG and CYP3A4 records
    assert any("hERG" in f for f in r["upadacitinib"]["flags"])  # no safety data: flagged, not "safe"
    assert r["filgotinib"]["metrics"]["delta_vs_JAK3"] == 0.53  # 7.25 - 6.72


def test_censored_bound_meets_floor_and_measured_value_converts():
    out = ria.run(EXT, BACKEND, KB)
    r = _by_name(out)
    assert r["tofacitinib"]["metrics"]["ic50_uM_hERG"] == ">30.0"
    assert math_close(r["filgotinib"]["metrics"]["ic50_uM_hERG"], 85.114, 0.01)  # 10**(9-4.07)/1000


def math_close(a, b, tol):
    return abs(a - b) <= tol


def test_missing_safety_data_policy_fail_rejects():
    p = copy.deepcopy(EXT)
    p["missing_data_policy"] = "fail"
    r = _by_name(ria.run(p, BACKEND, KB))
    assert r["upadacitinib"]["status"] == "rejected" and "safety_floor_hERG" in r["upadacitinib"]["reason"]
    assert r["filgotinib"]["status"] == "shortlisted"


def test_floor_above_censored_bound_is_not_a_pass():
    p = copy.deepcopy(EXT)
    for a in p["anti_targets"]:
        if a["gene"] == "hERG":
            a["min_ic50_uM"] = 200      # tofacitinib only known to be > 30 uM: cannot confirm >= 200
    p["missing_data_policy"] = "fail"
    r = _by_name(ria.run(p, BACKEND, KB))
    assert r["tofacitinib"]["status"] == "rejected"


def test_consistency_requires_exactly_one_antitarget_rule():
    p = copy.deepcopy(EXT)
    p["anti_targets"].append({"gene": "X", "min_ic50_uM": 5, "informational": True})
    assert not ria.check_consistency(p)["ok"]
    p2 = copy.deepcopy(EXT)
    p2["missing_data_policy"] = "maybe"
    assert not ria.check_consistency(p2)["ok"]


def test_rest_backend_pagination_retry_cache(tmp_path):
    calls = {"n": 0}

    class R:
        def __init__(self, js, code=200): self.js, self.status_code = js, code
        def raise_for_status(self): pass
        def json(self): return self.js

    class S:
        headers = {}
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return R({}, 503)                      # transient failure -> retry
            off = params.get("offset", 0)
            if off == 0:
                return R({"activities": [{"pchembl_value": "7.0", "standard_relation": "="}],
                          "page_meta": {"next": "x"}})
            return R({"activities": [{"pchembl_value": None, "standard_relation": ">", "standard_value": 30000,
                                      "standard_units": "nM"}], "page_meta": {"next": None}})

    b = ria.RestBackend(session=S(), cache_dir=tmp_path, sleep=lambda s: None)
    m = b.measurements("CHEMBL1", "CHEMBL240")
    assert m == {"values": [7.0], "censored": [[">", 30000.0]]}
    n_after = calls["n"]
    assert b.measurements("CHEMBL1", "CHEMBL240") == m and calls["n"] == n_after   # served from cache


def test_build_snapshot_roundtrip_with_fake_backend():
    from fullbody.snapshot import build_snapshot

    snap = build_snapshot(BACKEND, ["CHEMBL3301607"], ["JAK1", "hERG"])
    assert snap["molecules"]["CHEMBL3301607"]["pchembl"]["CHEMBL240"] == [4.07, 4.07]
    assert snap["molecules"]["CHEMBL3301607"]["censored"]["CHEMBL240"] == [[">", 100000]]
    try:
        build_snapshot(BACKEND, ["CHEMBL3301607"], ["NOPE"])
    except ValueError as e:
        assert "NOPE" in str(e)
    else:
        raise AssertionError("expected ValueError")
