"""Streamlit UI for fullbody-agent-poc. Run: streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullbody import ria  # noqa: E402
from fullbody.agent import simulate  # noqa: E402

SNAP = ROOT / "data/jak_snapshot.json"
snap = json.loads(SNAP.read_text(encoding="utf-8"))
names = {k: v["name"] for k, v in snap["molecules"].items()}

st.title("Full-body agent PoC (JAK / RA)")
st.warning("Research prototype on a pinned ChEMBL snapshot. Not clinical advice. Thresholds are unvalidated.")
tab1, tab2, tab3 = st.tabs(["Forward simulation", "Reverse inference", "Generate (hypotheses)"])

with tab1:
    mid = st.selectbox("Molecule", list(names), format_func=lambda k: names[k], key="mol")
    use_pk = st.checkbox("Use sourced PK (needs PK params for this molecule)", value=True)
    eff = st.slider("Efficacy occupancy threshold", 0.05, 0.9, 0.5, 0.05)
    if st.button("Simulate", key="sim"):
        s = simulate(mid, str(SNAP), use_pk=use_pk, thresholds={"occ_efficacy_min": eff})
        st.subheader(f"Verdict: {s['verdict']['decision']}")
        st.json(s["trace"])

with tab2:
    profs = sorted((ROOT / "profiles").glob("*.json"))
    pf = st.selectbox("Profile", profs, format_func=lambda p: p.name, index=len(profs) - 1, key="prof")
    if st.button("Rank candidates", key="rev"):
        prof = json.loads(Path(pf).read_text(encoding="utf-8"))
        out = ria.run(prof, ria.SnapshotBackend(str(SNAP)), ROOT / "knowledge/pathways.json")
        st.dataframe([{k: x.get(k) for k in ("rank", "name", "status", "reason", "flags")} for x in out["ranking"]])

with tab3:
    st.caption("BRICS recombination of reference fragments. No measured activity exists for these molecules.")
    top = st.number_input("Top N", 3, 30, 8)
    if st.button("Generate", key="gen"):
        from fullbody.generate import generate
        with st.spinner("Building and filtering..."):
            out = generate(SNAP, ROOT / "profiles/ra_jak1_safety.json", ROOT / "knowledge/pathways.json",
                           top_n=int(top), max_build=400)
        st.write("Funnel:", out["funnel"])
        st.dataframe([{"id": r["id"], "status": r["status"], "similarity_prior": r["similarity_prior"],
                       "smiles": r["smiles"], "qed": r["props"]["qed"], "sa": r["props"]["sa"]} for r in out["candidates"]])
        st.error(out["caveat"])
