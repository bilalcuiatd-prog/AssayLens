"""Export auditable, captioned research figures and HTML/JSON/CSV outputs."""
from __future__ import annotations
import html
import json
from pathlib import Path
import platform
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import sklearn
from .storage import dump_json


def generate_report(run_directory: Path, metrics: dict, residuals: pd.DataFrame, audit: dict, manifest: dict):
    import scipy
    about = {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    dump_json(run_directory / "environment.json", about)
    dump_json(run_directory / "manifest.json", manifest)
    dump_json(run_directory / "metrics.json", metrics)
    residuals.to_csv(run_directory / "test_residuals_anonymized.csv", index=False)
    # The held-out labels are recorded to enable assessment, not for candidate identification.
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 140})
    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    actual = residuals["actual"].to_numpy()
    lo = float(min(actual.min(), residuals.filter(like="prediction_").min().min()))
    hi = float(max(actual.max(), residuals.filter(like="prediction_").max().max()))
    for name, short in [("chemistry_only", "Chemistry"), ("chemistry_assay_weighted", "Assay + quality")]:
        ax.scatter(actual, residuals[f"prediction_{name}"], s=17, alpha=0.60, label=short)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Observed response (held-out)")
    ax.set_ylabel("Predicted response (held-out)")
    ax.set_title(f"Held-out predictions | {metrics['split_type']} split")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(run_directory / "observed_vs_predicted.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    names = ["Chemistry", "Chemistry + assay", "Assay + weighted"]
    keys = ["chemistry_only", "chemistry_assay", "chemistry_assay_weighted"]
    cov = [metrics["results"][key]["coverage"] for key in keys]
    ax.bar(names, cov)
    ax.axhline(1 - metrics["alpha"], color="black", linestyle="--", label="Nominal coverage")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out empirical coverage")
    ax.set_title("Calibration diagnostic (not a guarantee under shift)")
    ax.tick_params(axis="x", labelrotation=13)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(run_directory / "coverage.png", dpi=220)
    plt.close(fig)

    table_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{r['rmse']:.3f}</td><td>{r['mae']:.3f}</td>"
        f"<td>{r['r2']:.3f}</td><td>{r['coverage']:.3f}</td><td>{r['mean_interval_width']:.3f}</td></tr>"
        for name, r in metrics["results"].items()
    )
    caveats = "".join(f"<li>{html.escape(x)}</li>" for x in metrics["methodological_caveats"])
    warnings = "".join(f"<li>{html.escape(str(x))}</li>" for x in audit.get("warnings", []))
    demo_warning = ("<p style='background:#fff0cf;padding:13px;font-weight:bold;'>"
                    "SYNTHETIC DATA ONLY — DO NOT USE AS BIOLOGICAL OR PUBLICATION EVIDENCE.</p>") if audit.get("demo_data") else ""
    report = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>AssayLens AI research report</title><style>
body{{max-width:940px;margin:55px auto;font:16px/1.65 system-ui;color:#1d2731;padding:0 24px}}
h1{{font-size:30px;letter-spacing:-1px}}h2{{border-bottom:1px solid #d9e1e7;padding-bottom:8px;margin-top:35px}}
table{{width:100%;border-collapse:collapse;margin:18px 0}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #d9e1e7}}
th{{background:#f2f6f7}}img{{max-width:730px;width:100%;display:block;margin:20px auto}}
.small{{color:#596b77;font-size:13px}}code{{background:#f3f5f7;padding:3px}}</style></head><body>
<p class='small'>AssayLens AI · Automated computational evaluation · NOT peer-reviewed results</p>
<h1>Evidence-aware assay modeling: experimental report</h1>
{demo_warning}
<p><b>Dataset ID:</b> {html.escape(manifest['dataset_id'])}<br>
<b>Source SHA-256:</b> <code>{html.escape(manifest['source_sha256'])}</code><br>
<b>Evaluation:</b> {html.escape(metrics['split_type'])} groups, seed {metrics['seed']}, held-out n={metrics['test_n']}.
Calibration n={metrics['calibration_n']}; nominal coverage {1-metrics['alpha']:.0%}.</p>
<h2>Data integrity audit</h2>
<p>{audit['input_rows']} input records; {audit['curated_assay_compound_pairs']} curated pairs;
{audit['unique_scaffolds']} scaffolds; {audit['unique_assays']} assays.
Within-assay conflicts excluded: {audit['within_assay_conflict_pairs_excluded']}.</p>
<ul>{warnings}</ul>
<h2>Predefined model comparison</h2>
<table><thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Coverage</th><th>Width</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<img src='observed_vs_predicted.png' alt='Observed vs predicted'>
<img src='coverage.png' alt='Conformal coverage comparison'>
<h2>Methodological limitations</h2><ul>{caveats}</ul>
<h2>Reproducibility</h2><p>All source rows, curation decisions, training data, split indices,
model settings, environment versions and anonymized test predictions accompany the downloaded archive.
The application does not establish molecular efficacy, safety, clinical utility, or prospective assay performance.
Artificial demonstration data must never be used as paper evidence.</p>
<p class='small'>Generated by AssayLens AI. Verify all claims manually before using in a manuscript.</p>
</body></html>"""
    (run_directory / "report.html").write_text(report, encoding="utf-8")
    archive_path = run_directory / "research_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as z:
        for filename in ("report.html", "metrics.json", "environment.json", "manifest.json", "audit.json",
                         "test_residuals_anonymized.csv", "observed_vs_predicted.png", "coverage.png", "curated.csv", "qc_rows.csv", "raw.csv", "split_indices.json"):
            p = run_directory / filename
            if p.exists():
                z.write(p, arcname=filename)
    return archive_path
