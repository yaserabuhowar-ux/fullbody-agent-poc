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
        def pchembl(self, mol, tid):
            return [] if tid == self.data["targets"]["JAK2"] else super().pchembl(mol, tid)

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
    assert b.target_id("JAK1") == "CHEMBL9"
    assert b.candidates("CHEMBL9") == ["CHEMBL1"]
    assert b.pchembl("CHEMBL1", "CHEMBL9") == [7.5]
    assert b.properties("CHEMBL1")["qed"] == 0.8
    assert b.name("CHEMBL1") == "X"
