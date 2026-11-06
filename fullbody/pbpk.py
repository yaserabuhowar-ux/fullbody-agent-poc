"""Toy 5-compartment PBPK sketch (blood, liver, kidney, target tissue, rest of body).

Units: hours, L, L/h (inputs given in L/min and converted), mg, mg/L. All parameters are placeholders,
not calibrated to any drug. Mass balance is checked via mass_balance_error(). Not a validated simulation.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------- دعم العربية في الرسم (اختياري) ----------
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(s):
        return get_display(arabic_reshaper.reshape(s))
except ImportError:
    def ar(s):
        return s  # بدون الحزمتين ستظهر العربية مفككة: pip install arabic-reshaper python-bidi


class SimplifiedPBPK:
    """نموذج PBPK مبسط. الزمن بالساعات، التدفقات والتصفية بـ L/h، الكتلة بـ mg، التركيز mg/L.
    كل المعاملات افتراضية وتحتاج معايرة من بيانات حقيقية."""

    def __init__(self, dose_mg=400, k_abs=0.8, F=0.8, params=None):
        p = {
            # الأحجام (L)
            "V_blood": 5.0, "V_liver": 1.8, "V_kidney": 0.3, "V_target": 2.0, "V_other": 40.0,
            # التدفقات (L/min) -> تُحوَّل إلى L/h
            "Q_liver": 1.5, "Q_kidney": 1.2, "Q_target": 0.5, "Q_other": 2.0,
            # معاملات التقسيم
            "Kp_liver": 2.0, "Kp_kidney": 1.5, "Kp_target": 3.0, "Kp_other": 0.8,
            # التصفية (L/min) -> تُحوَّل إلى L/h
            "CL_int": 0.2, "CL_renal": 0.1,
        }
        if params:
            p.update(params)
        self.p = p
        self.k_abs, self.F, self.dose = k_abs, F, dose_mg
        for k in ("Q_liver", "Q_kidney", "Q_target", "Q_other", "CL_int", "CL_renal"):
            self.p[k] *= 60.0  # دقيقة -> ساعة
        self.Q_total = sum(self.p[k] for k in ("Q_liver", "Q_kidney", "Q_target", "Q_other"))

    def odes(self, t, y):
        A_gut, Cb, Cl, Ck, Ct, Co, E_cum, A_lost = y
        p = self.p

        absorbed = self.k_abs * A_gut                  # mg/h يغادر الأمعاء
        to_blood = absorbed * self.F                    # الجزء الواصل للدم
        lost = absorbed * (1 - self.F)                  # الجزء المفقود (first-pass/براز)

        Cvl, Cvk = Cl / p["Kp_liver"], Ck / p["Kp_kidney"]
        Cvt, Cvo = Ct / p["Kp_target"], Co / p["Kp_other"]

        dCb = (to_blood
               + p["Q_liver"] * Cvl + p["Q_kidney"] * Cvk
               + p["Q_target"] * Cvt + p["Q_other"] * Cvo
               - self.Q_total * Cb) / p["V_blood"]

        elim_liver = p["CL_int"] * Cvl                  # mg/h
        elim_kidney = p["CL_renal"] * Cvk               # mg/h

        dCl = (p["Q_liver"] * (Cb - Cvl) - elim_liver) / p["V_liver"]
        dCk = (p["Q_kidney"] * (Cb - Cvk) - elim_kidney) / p["V_kidney"]
        dCt = (p["Q_target"] * (Cb - Cvt)) / p["V_target"]
        dCo = (p["Q_other"] * (Cb - Cvo)) / p["V_other"]

        return [-absorbed, dCb, dCl, dCk, dCt, dCo, elim_liver + elim_kidney, lost]

    def simulate(self, t_span=(0, 72), n=500):
        y0 = [self.dose, 0, 0, 0, 0, 0, 0, 0]
        t_eval = np.linspace(*t_span, n)
        sol = solve_ivp(self.odes, t_span, y0, t_eval=t_eval, method="LSODA",
                        rtol=1e-8, atol=1e-10)
        return sol.t, sol.y

    def mass_balance_error(self, y):
        """أقصى انحراف عن الجرعة الكلية (mg). يجب أن يكون ≈ 0."""
        p = self.p
        A_gut, Cb, Cl, Ck, Ct, Co, E, lost = y
        in_body = (Cb * p["V_blood"] + Cl * p["V_liver"] + Ck * p["V_kidney"]
                   + Ct * p["V_target"] + Co * p["V_other"])
        total = A_gut + in_body + E + lost
        return np.abs(total - self.dose).max()


if __name__ == "__main__":
    model = SimplifiedPBPK(dose_mg=400)
    t, y = model.simulate()
    Cb, Cl, Ct = y[1], y[2], y[4]

    err = model.mass_balance_error(y)
    print(f"Mass balance error: {err:.2e} mg")
    assert err < 1e-4, "خرق حفظ الكتلة"

    for name, c in (("blood", Cb), ("target", Ct), ("liver", Cl)):
        print(f"Cmax {name}: {c.max():.3f} mg/L at t={t[c.argmax()]:.1f} h")

    plt.figure(figsize=(10, 6))
    plt.plot(t, Cb, label=ar("الدم"))
    plt.plot(t, Ct, label=ar("النسيج المستهدف"))
    plt.plot(t, Cl, label=ar("الكبد"))
    plt.xlabel(ar("الزمن (ساعة)"))
    plt.ylabel(ar("التركيز (mg/L)"))
    plt.title(ar("محاكاة PBPK مبسطة (معاملات افتراضية)"))
    plt.legend()
    plt.grid(True)
    plt.savefig("pbpk_result.png", dpi=150, bbox_inches="tight")
