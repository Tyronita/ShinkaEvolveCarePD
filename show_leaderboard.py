"""
Quick leaderboard viewer — run from ShinkaEvolveCarePD/:
    python show_leaderboard.py
    python show_leaderboard.py --top 20
    python show_leaderboard.py --csv path/to/other/leaderboard.csv
"""
import argparse
import csv
import json
import os

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "care_pd_task", "leaderboard.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"No leaderboard found at {args.csv}")
        print("Run evaluate.py or a shinka_run to generate results.")
        return

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Leaderboard is empty.")
        return

    # Sort by macro_f1 descending
    def _f1(r):
        try:
            return float(r.get("macro_f1", 0) or 0)
        except ValueError:
            return 0.0

    rows.sort(key=_f1, reverse=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_f1 = [_f1(r) for r in rows]
    print(f"\n{'='*72}")
    print(f"  CARE-PD Leaderboard  |  {len(rows)} genomes evaluated")
    print(f"{'='*72}")
    print(f"  Best macro-F1 : {max(all_f1):.4f}   (baseline: 0.5650)")
    print(f"  Mean macro-F1 : {sum(all_f1)/len(all_f1):.4f}")
    print(f"  Latest genome : {rows[-1]['genome_id'] if rows else 'N/A'}")
    print(f"{'='*72}\n")

    # ── Top-N table ───────────────────────────────────────────────────────────
    hdr = f"{'Rank':<5} {'Genome':<10} {'Macro-F1':>9} {'F1-0 (norm)':>11} {'F1-1 (mild)':>11} {'F1-2 (sev)':>10} {'Time(s)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for rank, row in enumerate(rows[: args.top], 1):
        gid   = row.get("genome_id", "?")
        f1    = _f1(row)
        f1_0  = row.get("f1_class0", "")
        f1_1  = row.get("f1_class1", "")
        f1_2  = row.get("f1_class2", "")
        t     = row.get("eval_time_s", "")
        arrow = " <-- BEST" if rank == 1 else ""
        print(f"{rank:<5} {gid:<10} {f1:>9.4f} {str(f1_0):>11} {str(f1_1):>11} {str(f1_2):>10} {str(t):>8}{arrow}")

    # ── Per-fold breakdown for the best genome ────────────────────────────────
    best = rows[0]
    fold_json = best.get("fold_f1_json", "")
    if fold_json:
        try:
            folds = json.loads(fold_json)
            print(f"\nPer-fold F1 for best genome ({best['genome_id']}):")
            for fold_id, f1_val in folds.items():
                print(f"  Fold {fold_id}: {f1_val:.4f}")
        except Exception:
            pass

    print()


if __name__ == "__main__":
    main()
