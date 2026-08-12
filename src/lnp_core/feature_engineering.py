"""Feature engineering for v5.3: strict one-hot alignment, Morgan FP with caching and warning suppression."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd

# Known lipid vocabulary
KNOWN_LIPID1 = ["ALC-0315", "SM-102"]
KNOWN_LIPID2 = ["DOPE", "DPPC", "DSPC"]
KNOWN_LIPID4 = ["DMG-PEG2000", "DSPE-PEG2000"]

# Design grid constants
RATIO1_LADDER = [40.0, 45.0, 50.0, 55.0, 60.0]
RATIO4_LADDER = [0.5, 1.0, 1.5, 2.0, 2.5]
RATIO2_FIXED = 10.0

# Morgan FP parameters
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
LIPID_SMILES_COLS = ["lipid1_smiles", "lipid2_smiles", "lipid3_smiles", "lipid4_smiles"]
RATIO_COLS = ["ratio1", "ratio2", "ratio3", "ratio4"]
CONTINUOUS_FEATURES = ["np_ratio", "aq_org_ratio"]


def design_one_hot_v5_3(df: pd.DataFrame) -> pd.DataFrame:
    """Build design-only one-hot features with strict lipid alignment.

    Ensures all known lipid types produce columns, even if absent in df.
    Raises ValueError if unknown lipid type encountered.

    Returns:
        DataFrame with one-hot encoded lipid columns + numeric features.
    """
    # Validate lipid types
    for col, known in [("lipid1", KNOWN_LIPID1), ("lipid2", KNOWN_LIPID2), ("lipid4", KNOWN_LIPID4)]:
        unknown = set(df[col].unique()) - set(known)
        if unknown:
            raise ValueError(f"Unknown {col} values: {unknown}. Expected subset of {known}")

    parts = []

    # One-hot for lipid1, lipid2, lipid4
    for col, known in [("lipid1", KNOWN_LIPID1), ("lipid2", KNOWN_LIPID2), ("lipid4", KNOWN_LIPID4)]:
        dummies = pd.get_dummies(df[col], prefix=col).astype(float)
        # Ensure all known columns present
        for val in known:
            cname = f"{col}_{val}"
            if cname not in dummies.columns:
                dummies[cname] = 0.0
        # Reorder to known order
        ordered_cols = [f"{col}_{v}" for v in known]
        parts.append(dummies[ordered_cols])

    # Numeric features
    for col in ["ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio"]:
        parts.append(pd.DataFrame({f"feat_{col}": df[col].astype(float).values}, index=df.index))

    # Polynomial features for ratio1
    parts.append(pd.DataFrame({"feat_ratio1_sq": (df["ratio1"] ** 2).astype(float).values}, index=df.index))
    parts.append(pd.DataFrame({"feat_ratio1_cu": (df["ratio1"] ** 3).astype(float).values}, index=df.index))

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result


@functools.lru_cache(maxsize=64)
def _compute_morgan_fp_cached(smiles: str, lipid_id: str = "") -> tuple[float, ...]:
    """Compute Morgan FP with LRU cache. Returns tuple for hashability."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not smiles or not isinstance(smiles, str):
            return tuple(np.zeros(MORGAN_NBITS, dtype=np.float64))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return tuple(np.zeros(MORGAN_NBITS, dtype=np.float64))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        arr = np.zeros(MORGAN_NBITS, dtype=np.float64)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
    return tuple(arr)


def design_morgan_fp_weighted_v5_3(df: pd.DataFrame) -> pd.DataFrame:
    """Build weighted Morgan FP features with caching and warning suppression.

    Algorithm:
    1. For each lipid role (1-4): for each unique SMILES, compute cached FP
    2. Weight by ratio_i / 100
    3. Append np_ratio, aq_org_ratio
    4. Total: 4 x 2048 + 2 = 8194 columns

    Returns:
        DataFrame, shape (n_samples, 8194)
    """
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    parts = []

    for smiles_col, ratio_col, role_name in zip(
        LIPID_SMILES_COLS, RATIO_COLS,
        ["lipid1", "lipid2", "lipid3", "lipid4"]
    ):
        # Compute FPs using cache
        fps = np.array([
            np.array(_compute_morgan_fp_cached(str(s), role_name))
            for s in df[smiles_col]
        ])
        weights = df[ratio_col].values.astype(np.float64) / 100.0
        weighted_fps = fps * weights[:, np.newaxis]
        fp_df = pd.DataFrame(
            weighted_fps,
            columns=[f"fp_{role_name}_{i}" for i in range(MORGAN_NBITS)],
            index=df.index,
        )
        parts.append(fp_df)

    # Continuous features
    for col in CONTINUOUS_FEATURES:
        parts.append(pd.DataFrame({f"feat_{col}": df[col].astype(float).values}, index=df.index))

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result
