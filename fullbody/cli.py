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
    f.add_argument("--pk", action="store_true", help="derive free concentration from the labelled dose regimen")
    f.add_argument("--occ-eff-min", type=float, default=None, help="override efficacy occupancy threshold (default 0.5, unvalidated)")
    f.add_argument("--snapshot", default=str(ROOT / "data/jak_snapshot.json"))
    v = sub.add_parser("validate", help="positive-control check: approved RA drugs under the PK model")
    v.add_argument("--snapshot", default=str(ROOT / "data/jak_snapshot.json"))
    r = sub.add_parser("reverse", help="reverse: rank candidates for a desired profile")
    r.add_argument("--profile", default=str(ROOT / "profiles/ra_jak1.json"))
    r.add_argument("--snapshot", default=str(ROOT / "data/jak_snapshot.json"))
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        print("Both drugs are approved and efficacious in RA; both are positive controls for the efficacy criterion.")
        for thr in (0.5, 0.15):
            print(f"\n-- efficacy occupancy threshold {thr}")
            for mid in ("CHEMBL3622821", "CHEMBL221959"):
                s = simulate(mid, a.snapshot, use_pk=True, thresholds={"occ_efficacy_min": thr})
                ar = s["body"]["at_regimen"]
                print(f"{s['snapshot']['molecules'][mid]['name']:<13} JAK1 mean occ {ar['efficacy_mean_occ']:<6} "
                      f"JAK2 mean occ {ar['liability_mean_occ']:<6} -> {s['verdict']['decision']}")
        print("\nNote: 0.15 is read off these same two drugs (circular). It shows the 0.5 default is "
              "inconsistent with approved efficacy, not that 0.15 is correct.")
        return 0
    if a.cmd == "simulate":
        s = simulate(a.molecule, a.snapshot, a.conc, use_pk=a.pk,
                     thresholds=({"occ_efficacy_min": a.occ_eff_min} if a.occ_eff_min else None))
        print(json.dumps({"trace": s["trace"], "verdict": s["verdict"]}, ensure_ascii=False, indent=2))
    else:
        prof = json.loads(Path(a.profile).read_text(encoding="utf-8"))
        out = ria.run(prof, ria.SnapshotBackend(a.snapshot), ROOT / "knowledge/pathways.json")
        for x in out["ranking"]:
            print(x["rank"] or "-", x["name"], x["status"], x.get("reason", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
