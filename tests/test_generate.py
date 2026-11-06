from pathlib import Path

from rdkit import Chem

from fullbody.generate import generate

ROOT = Path(__file__).resolve().parents[1]
ARGS = (ROOT / "data/jak_snapshot.json", ROOT / "profiles/ra_jak1_safety.json", ROOT / "knowledge/pathways.json")


def _run(**kw):
    return generate(*ARGS, top_n=20, max_build=300, **kw)


def test_deterministic():
    a, b = _run(), _run()
    assert [r["smiles"] for r in a["candidates"]] == [r["smiles"] for r in b["candidates"]]


def test_valid_novel_and_hypothesis_only():
    out = _run()
    assert out["candidates"], "expected at least one candidate"
    ref = {Chem.MolToInchiKey(Chem.MolFromSmiles(m["smiles"]))[:14]
           for m in __import__("json").loads(ARGS[0].read_text())["molecules"].values() if m.get("smiles")}
    for r in out["candidates"]:
        m = Chem.MolFromSmiles(r["smiles"])
        assert m is not None
        assert Chem.MolToInchiKey(m)[:14] not in ref          # seeds excluded
        assert r["status"] in ("hypothesis", "rejected")        # never "shortlisted"
        assert r["props"]["qed"] >= 0.5 and r["props"]["sa"] <= 4.5


def test_no_data_flags_present():
    for r in _run()["candidates"]:
        assert r["flags"], "generated molecules have no measurements; flags must say NO DATA"
        assert all("no data" in f.lower() for f in r["flags"])


def test_references_split_by_measured_selectivity():
    out = _run()
    assert "tofacitinib" in {v["name"] for v in out["references"]["non_selective"].values()}
    assert "upadacitinib" in {v["name"] for v in out["references"]["selective"].values()}
