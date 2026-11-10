"""Transparent molecular standardization and frozen feature extraction.

No compound design, generative synthesis, or candidate ranking is implemented.
"""
from __future__ import annotations
import hashlib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

FP_SIZE = 1024
FP_RADIUS = 2
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
DESCRIPTOR_NAMES = ["MolWt", "MolLogP", "HBD", "HBA", "TPSA", "RotBonds", "RingCount", "FractionCSP3"]


def parse_molecule(smiles: str) -> tuple[str, str, str, bool]:
    """Return canonical, largest-fragment parent, scaffold, and parent-changed flag."""
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 600:
        raise ValueError("Empty or overlong SMILES")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumHeavyAtoms() < 2:
        raise ValueError("Invalid or unsupported SMILES")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # Does not perform tautomer normalization: transformations are deliberately conservative.
    parent = rdMolStandardize.FragmentParent(mol)
    Chem.SanitizeMol(parent)  # FragmentParent may return an unsanitized mol (RingInfo not initialized).
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=True)
    # No Bemis-Murcko scaffold exists for acyclic structures. Keep exact structures grouped.
    if not scaf:
        scaf = "ACYCLIC:" + parent_smiles
    return canonical, parent_smiles, scaf, canonical != parent_smiles


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot fingerprint invalid SMILES: {smiles[:50]}")
    return GENERATOR.GetFingerprint(mol)


def feature_matrix(smiles: list[str]) -> np.ndarray:
    out = np.zeros((len(smiles), FP_SIZE + len(DESCRIPTOR_NAMES)), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid normalized SMILES in row {i}")
        fp = GENERATOR.GetFingerprint(mol)
        arr = np.zeros((FP_SIZE,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i, :FP_SIZE] = arr
        out[i, FP_SIZE:] = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        ]
    return out


def closest_tanimoto(queries: list[str], references: list[str]) -> np.ndarray:
    """Tanimoto-to-training for diagnostics, not activity-based ranking."""
    refs = [fingerprint(s) for s in sorted(set(references))]
    if not refs:
        raise ValueError("Reference set is empty")
    result = []
    for s in queries:
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint(s), refs)
        result.append(max(similarities))
    return np.asarray(result, dtype=float)


def anonymized_id(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf8")).hexdigest()[:12]
