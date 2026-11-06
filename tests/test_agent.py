import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullbody.agent import simulate  # noqa: E402

SNAP = ROOT / "data/jak_snapshot.json"


def test_upadacitinib_has_window_and_is_supported():
    s = simulate("CHEMBL3622821", SNAP, 100)
    assert s["verdict"]["decision"].startswith("SUPPORTED")
    assert s["organ"]["window_exists"] and s["organ"]["fold_selectivity"] == 14.11


def test_tofacitinib_blocked_by_safety_first_arbiter():
    s = simulate("CHEMBL221959", SNAP, 100)
    assert s["verdict"]["decision"] == "NOT SUPPORTED"
    assert not s["organ"]["window_exists"]


def test_occupancy_formula_hand_check():
    s = simulate("CHEMBL3622821", SNAP, 100)
    ic50 = s["target"]["JAK1"]["ic50_nM"]
    assert s["cell"]["IL-6/STAT3 (inflammation)"]["occupancy_at_conc"] == round(100 / (100 + ic50), 3)


def test_missing_structure_halts_to_inconclusive():
    s = simulate("CHEMBL2105759", SNAP, 100)  # baricitinib: no SMILES in snapshot
    assert s["verdict"]["decision"] == "INCONCLUSIVE"


def test_unknown_molecule_inconclusive():
    assert simulate("CHEMBL0", SNAP)["verdict"]["decision"] == "INCONCLUSIVE"


def test_rdkit_descriptors_match_chembl_mw():
    s = simulate("CHEMBL3622821", SNAP)
    assert abs(s["mol"]["xcheck"]["mw_diff"]) < 0.1
