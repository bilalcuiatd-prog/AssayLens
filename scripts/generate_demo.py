"""Creates entirely synthetic labels; never use as experimental scientific evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

rng = np.random.default_rng(42)
scaffolds = [
    "c1ccccc1", "c1ccncc1", "c1cnccn1", "c1ccoc1", "c1ccsc1",
    "c1ccc2ccccc2c1", "c1ccc2ncccc2c1", "C1CCCCC1", "C1CCCC1",
    "N1CCCCC1", "O1CCNCC1", "C1CCNCC1", "c1cc2ccccc2[nH]1",
    "c1cn[nH]c1", "c1ncc[nH]1", "c1cc2[nH]ccc2c1", "C1CCOC1",
]
prefixes = ["", "C", "CC", "CO", "CCC", "F", "Cl", "Br", "O", "N", "CN", "OC", "CCN", "COC", "CCO", "CCCl", "CCCO", "CNC", "CCF", "COC"]
assays = [f"SYNTH_ASSAY_{i:02d}" for i in range(1, 17)]
seen = set()
rows = []
for ring_id, core in enumerate(scaffolds):
    for prefix in prefixes:
        smi = prefix + core
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smi in seen:
            continue
        seen.add(smi)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        # Numeric values are fabricated from descriptors with random noise.
        intrinsic = 4.0 + 0.004 * mw + 0.12 * logp + ring_id * 0.055
        for replicate in range(int(rng.choice([1, 1, 2]))):
            assay_no = int(rng.integers(0, len(assays)))
            response = intrinsic + (assay_no % 4) * 0.18 + rng.normal(0, 0.2)
            rows.append({
                "smiles": smi,
                "response": round(float(response), 4),
                "assay_id": assays[assay_no],
                "assay_type": "B" if assay_no % 3 else "F",
                "assay_format": "biochemical" if assay_no % 2 else "cellular",
                "source_id": f"SYNTH_SOURCE_{assay_no % 3}",
                "endpoint": "SYNTHETIC_SIGNAL",
                "target_id": "SYNTHETIC_TARGET",
            })
frame = pd.DataFrame(rows)
frame.loc[len(frame)] = {**frame.iloc[0].to_dict(), "smiles": "not-a-smiles"}
frame.loc[len(frame)] = {**frame.iloc[1].to_dict(), "response": "missing"}
out = Path(__file__).resolve().parent.parent / "examples" / "synthetic_assay_demo.csv"
out.parent.mkdir(exist_ok=True, parents=True)
frame.to_csv(out, index=False)
print(f"Saved {len(frame)} synthetic measurements to {out}")
