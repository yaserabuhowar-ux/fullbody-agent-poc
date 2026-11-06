import argparse, json, sys
from pathlib import Path
from . import ria
from .agent import simulate

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fullbody")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("simulate", help="forward: run one molecule through the level agents")
    f.add_argument("molecule", help="ChEMBL id, e.g. CHEMBL3622821")
    f.add_argument("--conc", type=float, default=None, help="free concentration in nM")
    f.add_argument("--snapshot", default=str(ROOT / "data/jak_snapshot.json"))
    r = sub.add_parser("reverse", help="reverse: rank candidates for a desired profile")
    r.add_argument("--profile", default=str(ROOT / "profiles/ra_jak1.json"))
    r.add_argument("--snapshot", default=str(ROOT / "data/jak_snapshot.json"))
    a = ap.parse_args(argv)
    if a.cmd == "simulate":
        s = simulate(a.molecule, a.snapshot, a.conc)
        print(json.dumps({"trace": s["trace"], "verdict": s["verdict"]}, ensure_ascii=False, indent=2))
    else:
        prof = json.loads(Path(a.profile).read_text(encoding="utf-8"))
        out = ria.run(prof, ria.SnapshotBackend(a.snapshot), ROOT / "knowledge/pathways.json")
        for x in out["ranking"]:
            print(x["rank"] or "-", x["name"], x["status"], x.get("reason", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
