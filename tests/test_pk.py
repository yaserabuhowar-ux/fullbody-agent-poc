import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullbody import pk  # noqa: E402
from fullbody.agent import simulate  # noqa: E402

P = json.load(open(ROOT / "data/pk_params.json"))
SNAP = ROOT / "data/jak_snapshot.json"


def test_unit_conversion_hand_check():
    # 100 ng/mL of MW 400 = 250 nM total; fu 0.5 -> 125 nM
    assert math.isclose(pk.ng_ml_to_free_nM(100, 400, 0.5), 125.0)


def test_ka_solver_reproduces_tmax():
    ke = math.log(2) / 3.0
    ka = pk.solve_ka(1.0, ke)
    assert math.isclose(pk._tmax(ka, ke), 1.0, abs_tol=1e-3)


def test_upadacitinib_matches_label_auc_and_cmax_by_calibration():
    p = P["CHEMBL3622821"]
    m = pk.model_from_params(p)
    ts, prof = pk.steady_state_profile(15, 24, m)
    assert math.isclose(pk.summarize(ts, prof)["auc_tau_ng_h_ml"], 265.0, rel_tol=0.01)  # AUC ss = AUCinf single
    single = pk._single_cmax(15, m["ke"], m["ka"], m["f_over_v"])
    assert math.isclose(single, 31.6, rel_tol=0.02)


def test_tofacitinib_bracketed_by_reported_auc_values():
    p = P["CHEMBL221959"]
    m = pk.model_from_params(p)
    ts, prof = pk.steady_state_profile(5, 12, m)
    auc24 = 2 * pk.summarize(ts, prof)["auc_tau_ng_h_ml"]
    assert 263.4 < auc24 < 504   # between label Table 8 and RA Table 9
    assert abs(pk.summarize(ts, prof)["cmax_ng_ml"] - 42.7) / 42.7 < 0.25


def test_every_pk_parameter_has_a_source():
    for mid, p in P.items():
        if mid.startswith("_"):
            continue
        for k, v in p.items():
            if isinstance(v, dict) and "value" in v:
                assert v.get("source"), (mid, k)


def test_documented_validation_behaviour():
    """At the default 0.5 threshold both approved drugs fail efficacy; at 0.15 selectivity separates them."""
    for mid in ("CHEMBL3622821", "CHEMBL221959"):
        assert simulate(mid, SNAP, use_pk=True)["verdict"]["decision"] == "NOT SUPPORTED"
    up = simulate("CHEMBL3622821", SNAP, use_pk=True, thresholds={"occ_efficacy_min": 0.15})
    tof = simulate("CHEMBL221959", SNAP, use_pk=True, thresholds={"occ_efficacy_min": 0.15})
    assert up["verdict"]["decision"].startswith("SUPPORTED") and tof["verdict"]["decision"] == "NOT SUPPORTED"


def test_pk_missing_params_halts():
    s = simulate("CHEMBL3301607", SNAP, use_pk=True)  # filgotinib: no sourced PK parameters -> halts
    assert s["verdict"]["decision"] == "INCONCLUSIVE"
    assert any("no sourced PK parameters" in t["finding"] for t in s["trace"])
