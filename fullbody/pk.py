"""One-compartment oral PK with first-order absorption and superposition for repeated dosing.

C(t) = (F*D*ka / (V*(ka-ke))) * (exp(-ke t) - exp(-ka t))     [F/V or CL/F forms]
ka is solved numerically so that the model Tmax equals the sourced Tmax.
This is a population-mean, single-compartment sketch: no inter-individual variability, no
nonlinearity, no food/formulation effects (ER release is lumped into ka).
"""
from __future__ import annotations

import math


def _tmax(ka: float, ke: float) -> float:
    return math.log(ka / ke) / (ka - ke)


def solve_ka(tmax_h: float, ke: float) -> float:
    lo, hi = ke * 1.0001, 200.0  # tmax decreases as ka increases
    for _ in range(200):
        mid = (lo + hi) / 2
        if _tmax(mid, ke) > tmax_h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _single_cmax(dose_mg: float, ke: float, ka: float, f_over_v: float) -> float:
    m = {"ke": ke, "ka": ka, "f_over_v": f_over_v}
    return max(conc_ng_ml(t / 20, dose_mg, m) for t in range(0, 20 * 96))


def model_from_params(p: dict) -> dict:
    tmax = p["tmax_h"]["value"]
    if "CL_F_L_h" in p:  # apparent parameters
        cl_f = p["CL_F_L_h"]["value"]
        target = p.get("reported_single_dose", {}).get("cmax_ng_ml")
        dose = p["regimen"]["dose_mg"]
        if target:  # fit effective t1/2 so single-dose Cmax matches the label (AUC fixed by CL/F)
            lo, hi = 1.0, 14.0  # Cmax rises as t1/2 falls
            for _ in range(60):
                mid = (lo + hi) / 2
                ke = math.log(2) / mid
                cmax = _single_cmax(dose, ke, solve_ka(tmax, ke), ke / cl_f)
                lo, hi = (mid, hi) if cmax > target else (lo, mid)
            t_half = (lo + hi) / 2
            fitted = True
        else:
            t_half, fitted = p["t_half_h"]["value"], False
        ke = math.log(2) / t_half
        f_over_v = ke / cl_f  # 1/(V/F)
    else:
        t_half, fitted = p["t_half_h"]["value"], False
        ke = math.log(2) / t_half
        f_over_v = p["F"]["value"] / p["V_L"]["value"]
    return {"ke": ke, "ka": solve_ka(tmax, ke), "f_over_v": f_over_v, "t_half_used_h": t_half,
            "fitted_to_label_cmax": fitted}


def conc_ng_ml(t_h: float, dose_mg: float, m: dict) -> float:
    """Total plasma concentration (ng/mL) after a single dose at t=0."""
    if t_h < 0:
        return 0.0
    ka, ke = m["ka"], m["ke"]
    coef = dose_mg * m["f_over_v"] * ka / (ka - ke)  # mg/L = ug/mL
    return 1000.0 * coef * (math.exp(-ke * t_h) - math.exp(-ka * t_h))


def steady_state_profile(dose_mg: float, interval_h: float, m: dict, n_points: int = 241, n_prior: int = 40):
    ts = [interval_h * i / (n_points - 1) for i in range(n_points)]
    prof = [sum(conc_ng_ml(t + k * interval_h, dose_mg, m) for k in range(n_prior)) for t in ts]
    return ts, prof


def summarize(ts, prof):
    tau = ts[-1] - ts[0]
    auc = sum((prof[i] + prof[i + 1]) / 2 * (ts[i + 1] - ts[i]) for i in range(len(ts) - 1))
    return {"cmax_ng_ml": max(prof), "cmin_ng_ml": prof[0], "auc_tau_ng_h_ml": auc, "cavg_ng_ml": auc / tau,
            "tmax_h": ts[prof.index(max(prof))]}


def ng_ml_to_free_nM(c_ng_ml: float, mw: float, fu: float) -> float:
    return c_ng_ml * 1000.0 / mw * fu  # 1 ng/mL = 1000/MW nmol/L
