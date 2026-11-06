"""Build a pinned snapshot JSON from any ChEMBL backend (live RestBackend on a machine with network)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def build_snapshot(backend, molecule_ids: list[str], genes: list[str], smiles: dict | None = None) -> dict:
    tids = {g: backend.target_id(g) for g in genes}
    unresolved = [g for g, t in tids.items() if t is None]
    if unresolved:
        raise ValueError(f"unresolved genes: {unresolved}")
    mols = {}
    for m in molecule_ids:
        rec = {"name": backend.name(m), "properties": backend.properties(m), "pchembl": {}, "censored": {}}
        for g, tid in tids.items():
            meas = backend.measurements(m, tid)
            if meas["values"]:
                rec["pchembl"][tid] = meas["values"]
            if meas["censored"]:
                rec["censored"][tid] = meas["censored"]
        if smiles and m in smiles:
            rec["smiles"] = smiles[m]
        mols[m] = rec
    return {"meta": {"source": f"ChEMBL REST via {type(backend).__name__}",
                     "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     "sampling": "all IC50 records returned (paginated), mixed assay formats"},
            "targets": tids, "molecules": mols}


def write_snapshot(snap: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(snap, indent=1, ensure_ascii=False), encoding="utf-8")
