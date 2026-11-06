# Full-Body Agent PoC

Supervisor agent + level agents (LangGraph) that run a molecule through molecule → target → cell → organ → body checks, and a reverse loop that ranks ChEMBL candidates against a desired profile. Data: ChEMBL v34 snapshot (JAK1/JAK2).

```bash
pip install rdkit langgraph pytest
python -m fullbody.cli simulate CHEMBL3622821 --conc 100   # upadacitinib -> SUPPORTED (window 42.7-150.7 nM free)
python -m fullbody.cli simulate CHEMBL221959 --conc 100    # tofacitinib  -> NOT SUPPORTED (no JAK1-over-JAK2 window)
python -m fullbody.cli simulate CHEMBL3622821 --pk          # dose -> free-concentration profile -> occupancy
python -m fullbody.cli validate                             # positive-control check (see below)
python -m fullbody.cli reverse --profile profiles/ra_jak1.json
pytest -q
```

## Architecture
`decompose -> molecule (RDKit) -> target (ChEMBL IC50) -> cell (occupancy) -> organ (efficacy vs liability) -> body (window) -> arbiter (safety-first, with trace chain)`.
Any level that lacks data halts the graph to an INCONCLUSIVE verdict instead of guessing.

## PK agent (v0.2)
One-compartment oral model with first-order absorption (`fullbody/pk.py`). Inputs in `data/pk_params.json`, each with its source:
tofacitinib 5 mg q12h (F 74%, V 87 L, t1/2 ~3 h, 40% protein-bound; XELJANZ label), upadacitinib ER 15 mg q24h
(CL/F derived from label AUCinf 265 ng·h/mL; protein binding 52% from a secondary source, not the label).

Checks against label values:
- tofacitinib: model Cmax 36.2 vs label 42.7 ng/mL; model AUC24 368 sits between label Table 8 (263) and RA Table 9 (504).
- upadacitinib: a plain terminal t1/2 (11 h) underpredicted single-dose Cmax by ~56% (13.8 vs 31.6), so the effective t1/2 was FITTED (2.66 h) to the label Cmax. Steady-state trough is therefore not predictive.

**Known failure:** with the default efficacy threshold (JAK1 mean occupancy >= 0.5 against enzyme IC50 and free plasma concentration) both approved, efficacious RA drugs come out NOT SUPPORTED (mean occupancy ~0.2). The threshold is unvalidated; `fullbody validate` shows this. Lowering it to 0.15 separates the two drugs through JAK2 liability, but that value was read off the same two drugs, so it is circular, not a validation. Enzyme IC50 vs free plasma exposure ignores cellular potency and tissue distribution.

## What this is not
- Not a validated physiological simulation: occupancy = C/(C+IC50) ignores ATP competition, protein binding, PK, tissue distribution.
- Levels 2, 4, 6 and the hormonal/mineral/neural layers are not implemented.
- IC50 data are the first <=5 records per molecule/target from mixed ChEMBL assays; spreads can be wide (e.g. upadacitinib JAK2 pChEMBL 6.22-8.1), so the window is a screening signal only.
- Pathway/target map (JAK1 = efficacy, JAK2 = marrow liability) is hand-curated seed knowledge needing expert review.
- Live ChEMBL REST backend (`fullbody/ria.py: RestBackend`) is untested against the network.
- Only upadacitinib and tofacitinib have structures in the snapshot.

## Update: live data path, safety profile, generator, UI

**Data.** `RestBackend` (pagination, retry/backoff, on-disk cache) fetches ChEMBL live; `python -m fullbody.cli snapshot --molecules ...` pins a snapshot. **The live REST path has only been tested against a mocked session** (the build sandbox could not reach ChEMBL). The shipped snapshot was built from the ChEMBL MCP tools and covers the first N records per pair, not a random sample.

**Safety profile** (`profiles/ra_jak1_safety.json`): JAK2/JAK3 selectivity deltas, hERG and CYP3A4 IC50 floors (10 uM, a screening assumption, not a validated margin), TYK2 informational. Missing data follows `missing_data_policy` (`fail`/`pass`/`flag`); `flag` passes but records NO DATA. Censored ">X" values count only if X is at or above the floor.

**Generator** (`python -m fullbody.cli generate`): BRICS recombination of reference-drug fragments, filtered by heavy-atom count, Ro5, QED, PAINS/Brenk, SA score, and Tanimoto range. Output is **hypothesis only**: generated molecules have no measured activity, so potency/selectivity/safety are NO DATA flags. The ranking prior is similarity to selective vs non-selective references; it is a heuristic, not an activity prediction.

**UI:** `pip install streamlit` then `streamlit run app/streamlit_app.py`.

**Known limits:** unvalidated occupancy thresholds (0.5 default fails both approved RA drugs); upadacitinib PK is calibrated to label Cmax so trough is unreliable; 5 reference drugs only.
