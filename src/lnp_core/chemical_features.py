"""Chemical feature generation using Morgan fingerprints for v5.1."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Morgan FP parameters
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
LIPID_SMILES_COLS = ["lipid1_smiles", "lipid2_smiles", "lipid3_smiles", "lipid4_smiles"]
RATIO_COLS = ["ratio1", "ratio2", "ratio3", "ratio4"]
CONTINUOUS_FEATURES = ["np_ratio", "aq_org_ratio"]


def compute_morgan_fp(
    smiles: str,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
) -> np.ndarray:
    """Compute Morgan fingerprint for a single SMILES string.

    Args:
        smiles: SMILES string
        radius: Morgan FP radius (default 2)
        n_bits: Output bit count (default 2048)

    Returns:
        np.ndarray of shape (n_bits,), dtype float64

    Fallback: if SMILES is invalid (RDKit cannot parse), returns zero vector.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if not smiles or not isinstance(smiles, str):
        return np.zeros(n_bits, dtype=np.float64)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float64)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float64)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    return arr


def build_morgan_fp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build formulation-level Morgan FP feature matrix.

    Algorithm:
    1. For each lipid role (1-4), compute Morgan FP (2048 bits)
    2. Weight each role's FP by ratio_i / 100
    3. Concatenate all 4 weighted FPs -> 4 x 2048 = 8192 columns
    4. Append np_ratio, aq_org_ratio -> 8194 columns

    Returns:
        DataFrame, shape (n_samples, 8194)
    """
    parts = []

    for smiles_col, ratio_col in zip(LIPID_SMILES_COLS, RATIO_COLS):
        # Compute FPs for all SMILES in this role
        fps = np.array([compute_morgan_fp(s) for s in df[smiles_col]])
        # Weight by ratio
        weights = df[ratio_col].values.astype(np.float64) / 100.0
        weighted_fps = fps * weights[:, np.newaxis]
        role_tag = smiles_col.replace("_smiles", "")
        fp_df = pd.DataFrame(
            weighted_fps,
            columns=[f"fp_{role_tag}_{i}" for i in range(MORGAN_NBITS)],
            index=df.index,
        )
        parts.append(fp_df)

    # Continuous features
    for col in CONTINUOUS_FEATURES:
        parts.append(pd.DataFrame({f"feat_{col}": df[col].astype(float).values}, index=df.index))

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result


def validate_smiles_coverage(df: pd.DataFrame) -> dict:
    """Validate SMILES column coverage.

    Returns:
        dict mapping column name to {"valid": N, "invalid": M, "total": T}
    """
    from rdkit import Chem

    result = {}
    for col in LIPID_SMILES_COLS:
        total = len(df)
        valid = 0
        for s in df[col]:
            if isinstance(s, str) and Chem.MolFromSmiles(s) is not None:
                valid += 1
        result[col] = {"valid": valid, "invalid": total - valid, "total": total}
    return result
