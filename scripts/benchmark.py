"""Pre-specified repeated hold-out comparison. Run separately from the UI.

Usage: python scripts/benchmark.py --csv your_data.csv --splits scaffold assay \
       --seeds 3 13 23 33 43 --trees 96 --output benchmark_results.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assaylens.quality import audit_dataset
from assaylens.modeling import train_evaluate


def main():
    p = argparse.ArgumentParser(description="AssayLens repeated, group-disjoint regression benchmark")
    p.add_argument("--csv", required=True)
    p.add_argument("--splits", nargs="+", choices=["scaffold", "assay"], default=["scaffold", "assay"])
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])
    p.add_argument("--trees", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--output", default="benchmark_results.csv")
    args = p.parse_args()
    audited = audit_dataset(pd.read_csv(args.csv))
    results = []
    errors = []
    for mode in args.splits:
        for seed in args.seeds:
            try:
                run = train_evaluate(audited.curated, split_type=mode, seed=seed, trees=args.trees, alpha=args.alpha)
                for variant, metric in run.summary["results"].items():
                    results.append({"split": mode, "seed": seed, "variant": variant, **metric})
                print(f"OK: {mode} / seed {seed}")
            except ValueError as exc:
                errors.append(f"{mode}/seed {seed}: {exc}")
                print("SKIP:", errors[-1])
    if not results:
        raise SystemExit("No successful splits. Inspect assay group sizes and dataset quality.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output, index=False)
    numeric = ["rmse", "mae", "r2", "coverage", "mean_interval_width", "ood_coverage", "in_domain_coverage", "ood_mae"]
    summary = df.groupby(["split", "variant"])[numeric].agg(["mean", "std", "count"]).round(4)
    summary.to_csv(output.with_suffix(".summary.csv"))
    manifest = {"source": args.csv, "seeds": args.seeds, "splits": args.splits,
                "models": ["chemistry_only", "chemistry_assay", "chemistry_assay_weighted"],
                "errors": errors, "audit": audited.summary}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(summary.to_string())
    print("Data are not evidence of generalization without external and prospective validation.")


if __name__ == "__main__":
    main()
