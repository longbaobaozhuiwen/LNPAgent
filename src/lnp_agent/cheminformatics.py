"""组合化学引擎: Ugi-3CR 虚拟库生成 + 脂质组合库生成。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolEnumerator

logger = logging.getLogger(__name__)

# 抑制 RDKit 警告
RDLogger.logger().setLevel(RDLogger.ERROR)

# Ugi-3CR 核心: 4 个原子, 3 个连接位 (0=胺, 1=醛, 3=异腈)
UGI3CR_CORE_SMILES = "NCC(N)=O"
UGI3CR_CORE_POSITIONS = {"A": 0, "B": 1, "C": 3}

# 反应基团 SMARTS
FRAG_NH2 = Chem.MolFromSmarts("[NH2]")
FRAG_CHO = Chem.MolFromSmarts("[CX3H1](=O)")
FRAG_NC = Chem.MolFromSmarts("N#C")
WILDCARD = Chem.MolFromSmiles("*")


@dataclass
class VirtualLibraryResult:
    """虚拟库生成结果。"""
    library_df: pd.DataFrame
    n_total: int
    n_valid: int
    n_invalid: int
    generation_mode: str


def validate_smiles(smiles: str) -> tuple[bool, Chem.Mol | None]:
    """验证 SMILES 字符串。返回 (is_valid, mol)。"""
    if not smiles or not isinstance(smiles, str):
        return False, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None, mol
    except Exception:
        return False, None


def _strip_reactive_group(mol: Chem.Mol, frag: Chem.Mol) -> Chem.Mol | None:
    """从分子中剥离反应基团，替换为通配符 *。"""
    try:
        replaced = Chem.ReplaceSubstructs(mol, frag, WILDCARD)
        if replaced:
            return replaced[0]
    except Exception:
        pass
    return None


def assemble_ugi3cr_molecule(
    a_smiles: str,
    b_smiles: str,
    c_smiles: str,
    core_smiles: str = UGI3CR_CORE_SMILES,
) -> tuple[bool, str]:
    """组装单个 Ugi-3CR 产物。

    算法 (来自 AGILE notebook):
    1. 剥离各组件的反应基团 → * 通配符
    2. 计算原子位置映射
    3. 构建 SRU 分支标记组合 SMILES
    4. 用 rdMolEnumerator.Enumerate 解析

    Returns:
        (success, product_smiles)
    """
    # 验证输入
    valid_a, mol_a = validate_smiles(a_smiles)
    valid_b, mol_b = validate_smiles(b_smiles)
    valid_c, mol_c = validate_smiles(c_smiles)
    if not (valid_a and valid_b and valid_c):
        return False, ""

    # 剥离反应基团
    stripped_a = _strip_reactive_group(mol_a, FRAG_NH2)
    stripped_b = _strip_reactive_group(mol_b, FRAG_CHO)
    stripped_c = _strip_reactive_group(mol_c, FRAG_NC)
    if stripped_a is None or stripped_b is None or stripped_c is None:
        return False, ""

    smiles_a = Chem.MolToSmiles(stripped_a)
    smiles_b = Chem.MolToSmiles(stripped_b)
    smiles_c = Chem.MolToSmiles(stripped_c)

    if not (smiles_a.startswith("*") and smiles_b.startswith("*") and smiles_c.startswith("*")):
        return False, ""

    # 计算位置映射
    core_mol = Chem.MolFromSmiles(core_smiles)
    core_num_atoms = core_mol.GetNumAtoms()
    star_a_pos = core_num_atoms
    star_b_pos = core_num_atoms + stripped_a.GetNumAtoms()
    star_c_pos = core_num_atoms + stripped_a.GetNumAtoms() + stripped_b.GetNumAtoms()

    # 构建 SRU 组合 SMILES
    combined_smiles = (
        f"{core_smiles}.{smiles_a}.{smiles_b}.{smiles_c}"
        f" |m:{star_a_pos}:{UGI3CR_CORE_POSITIONS['A']},"
        f"{star_b_pos}:{UGI3CR_CORE_POSITIONS['B']},"
        f"{star_c_pos}:{UGI3CR_CORE_POSITIONS['C']}|"
    )

    try:
        mol_to_combine = Chem.MolFromSmiles(combined_smiles)
        if mol_to_combine is None:
            return False, ""
        enumerated = rdMolEnumerator.Enumerate(mol_to_combine)
        if not enumerated:
            return False, ""
        product_smiles = Chem.MolToSmiles(enumerated[0])
        return True, product_smiles
    except Exception as e:
        logger.debug(f"Ugi-3CR assembly failed: {e}")
        return False, ""


def generate_ugi3cr_library(
    a_smiles_list: list[str],
    b_smiles_list: list[str],
    c_smiles_list: list[str],
    core_smiles: str = UGI3CR_CORE_SMILES,
) -> VirtualLibraryResult:
    """完整 Ugi-3CR 组合库生成 (笛卡尔积)。

    Returns:
        VirtualLibraryResult with columns:
        id, label, combined_smiles, a_smiles, b_smiles, c_smiles, is_valid
    """
    records = []
    count = 0
    n_valid = 0
    n_invalid = 0

    for i, a in enumerate(a_smiles_list):
        for j, b in enumerate(b_smiles_list):
            for k, c in enumerate(c_smiles_list):
                success, product = assemble_ugi3cr_molecule(a, b, c, core_smiles)
                if success:
                    n_valid += 1
                else:
                    n_invalid += 1
                    product = ""
                records.append({
                    "id": count,
                    "label": f"A{i+1}B{j+1}C{k+1}",
                    "combined_smiles": product,
                    "a_smiles": a,
                    "b_smiles": b,
                    "c_smiles": c,
                    "is_valid": success,
                })
                count += 1

    df = pd.DataFrame(records)
    return VirtualLibraryResult(
        library_df=df,
        n_total=count,
        n_valid=n_valid,
        n_invalid=n_invalid,
        generation_mode="ugi3cr",
    )


def generate_lipid_combination_library(
    ionizable_smiles: list[dict],
    helper_smiles: list[dict],
    peg_smiles: list[dict],
    ratio1_ladder: list[float] | None = None,
    ratio4_ladder: list[float] | None = None,
    np_ratio_options: list[int] | None = None,
) -> VirtualLibraryResult:
    """脂质角色组合库生成 (无化学反应)。

    Args:
        ionizable_smiles: [{"name": str, "smiles": str}, ...]
        helper_smiles: 同上
        peg_smiles: 同上
        ratio1_ladder: ratio1 候选值列表 (默认 [46.3, 50.0])
        ratio4_ladder: ratio4 候选值列表 (默认 [1.5, 1.6])
        np_ratio_options: np_ratio 候选值列表 (默认 [3, 6])

    Returns:
        VirtualLibraryResult with formulation-ready DataFrame:
        lipid1-4, ratio1-4, np_ratio, aq_org_ratio, *_smiles
    """
    if ratio1_ladder is None:
        ratio1_ladder = [46.3, 50.0]
    if ratio4_ladder is None:
        ratio4_ladder = [1.5, 1.6]
    if np_ratio_options is None:
        np_ratio_options = [3, 6]

    records = []
    count = 0

    for ion in ionizable_smiles:
        for hlp in helper_smiles:
            for peg in peg_smiles:
                for r1 in ratio1_ladder:
                    for r4 in ratio4_ladder:
                        for npr in np_ratio_options:
                            r3 = 100.0 - r1 - 10.0 - r4  # cholesterol 固定 10%
                            records.append({
                                "lipid1": ion["name"],
                                "lipid2": hlp["name"],
                                "lipid3": "Cholesterol",
                                "lipid4": peg["name"],
                                "ratio1": r1,
                                "ratio2": 10.0,
                                "ratio3": round(r3, 1),
                                "ratio4": r4,
                                "np_ratio": npr,
                                "aq_org_ratio": 3.0,
                                "lipid1_smiles": ion["smiles"],
                                "lipid2_smiles": hlp["smiles"],
                                "lipid3_smiles": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
                                "lipid4_smiles": peg["smiles"],
                            })
                            count += 1

    df = pd.DataFrame(records)
    return VirtualLibraryResult(
        library_df=df,
        n_total=count,
        n_valid=count,
        n_invalid=0,
        generation_mode="lipid_combination",
    )
