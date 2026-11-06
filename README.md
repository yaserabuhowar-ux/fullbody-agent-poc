# Full-Body Agent PoC

Supervisor agent + level agents (LangGraph) that run a molecule through molecule → target → cell → organ → body checks, and a reverse loop that ranks ChEMBL candidates against a desired profile. Data: ChEMBL v34 snapshot (JAK1/JAK2).

```bash
pip install rdkit langgraph pytest
python -m fullbody.cli simulate CHEMBL3622821 --conc 100   # upadacitinib -> SUPPORTED (window 42.7-150.7 nM free)
python -m fullbody.cli simulate CHEMBL221959 --conc 100    # tofacitinib  -> NOT SUPPORTED (no JAK1-over-JAK2 window)
python -m fullbody.cli reverse --profile profiles/ra_jak1.json
pytest -q
```

## Architecture
`decompose -> molecule (RDKit) -> target (ChEMBL IC50) -> cell (occupancy) -> organ (efficacy vs liability) -> body (window) -> arbiter (safety-first, with trace chain)`.
Any level that lacks data halts the graph to an INCONCLUSIVE verdict instead of guessing.

## What this is not
- Not a validated physiological simulation: occupancy = C/(C+IC50) ignores ATP competition, protein binding, PK, tissue distribution.
- Levels 2, 4, 6 and the hormonal/mineral/neural layers are not implemented.
- IC50 data are the first <=5 records per molecule/target from mixed ChEMBL assays; spreads can be wide (e.g. upadacitinib JAK2 pChEMBL 6.22-8.1), so the window is a screening signal only.
- Pathway/target map (JAK1 = efficacy, JAK2 = marrow liability) is hand-curated seed knowledge needing expert review.
- Live ChEMBL REST backend (`fullbody/ria.py: RestBackend`) is untested against the network.
- Only upadacitinib and tofacitinib have structures in the snapshot.
