from fullbody.pbpk import SimplifiedPBPK


def test_mass_balance():
    m = SimplifiedPBPK(dose_mg=400)
    t, y = m.simulate()
    assert m.mass_balance_error(y) < 1e-6


def test_no_negative_concentrations_and_gut_empties():
    m = SimplifiedPBPK(dose_mg=400)
    t, y = m.simulate()
    assert y[:6].min() > -1e-9
    assert y[0][-1] < 1e-6 * 400


def test_units_clearance_scales_with_hours():
    # doubling hepatic clearance must lower the blood AUC
    base = SimplifiedPBPK(dose_mg=400)
    fast = SimplifiedPBPK(dose_mg=400, params={"CL_int": 0.4})
    tb, yb = base.simulate()
    tf, yf = fast.simulate()
    assert yf[1].sum() < yb[1].sum()
