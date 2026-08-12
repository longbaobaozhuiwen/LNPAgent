"""策展构建块库: Ugi-3CR + Lipid Combination 模式。

数据来源:
- AGILE Ugi-3CR 数据集 (1,200 产物, 20 amines × 12 aldehydes × 5 isocyanides)
- 文献策展的 ionizable / helper / PEG 脂质
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════
# Ugi-3CR 构建块 (来自 AGILE 数据集)
# ═══════════════════════════════════════════════════════════

AGILE_AMINES: list[dict] = [
    {"name": "amine_01", "smiles": "CN(C)CCN", "source": "AGILE"},
    {"name": "amine_02", "smiles": "NCCCN1CCCC1", "source": "AGILE"},
    {"name": "amine_03", "smiles": "CCN(CC)CCCN", "source": "AGILE"},
    {"name": "amine_04", "smiles": "CCN(CCN)CC", "source": "AGILE"},
    {"name": "amine_05", "smiles": "NCCN(CCN)CCN", "source": "AGILE"},
    {"name": "amine_06", "smiles": "CN(C)CCCN", "source": "AGILE"},
    {"name": "amine_07", "smiles": "NCCCN1CCOCC1", "source": "AGILE"},
    {"name": "amine_08", "smiles": "CCCCN(CCCC)CCCN", "source": "AGILE"},
    {"name": "amine_09", "smiles": "CC1=NN(C)C(N)=C1", "source": "AGILE"},
    {"name": "amine_10", "smiles": "NCCN1CCCC1", "source": "AGILE"},
    {"name": "amine_11", "smiles": "NN1CCOCC1", "source": "AGILE"},
    {"name": "amine_12", "smiles": "CN1CCC(N)CC1", "source": "AGILE"},
    {"name": "amine_13", "smiles": "NN1CCCCC1", "source": "AGILE"},
    {"name": "amine_14", "smiles": "NC1=CNN=C1", "source": "AGILE"},
    {"name": "amine_15", "smiles": "NCCN1CCCCC1", "source": "AGILE"},
    {"name": "amine_16", "smiles": "CC(N(C(C)C)CCN)C", "source": "AGILE"},
    {"name": "amine_17", "smiles": "NCCCN(CCCN)C", "source": "AGILE"},
    {"name": "amine_18", "smiles": "CCN1C(CN)CCC1", "source": "AGILE"},
    {"name": "amine_19", "smiles": "NCCN1CCNCC1", "source": "AGILE"},
    {"name": "amine_20", "smiles": "NC1CCNCC1", "source": "AGILE"},
]

AGILE_ALDEHYDES: list[dict] = [
    {"name": "aldehyde_01", "smiles": "O=C(CCCCCCCC)OCCCCCC=O", "source": "AGILE"},
    {"name": "aldehyde_02", "smiles": "CC(CCC(OCCCCCC=O)=O)CCCCC", "source": "AGILE"},
    {"name": "aldehyde_03", "smiles": "O=C(CCCCCCCCC)OCCCCCC=O", "source": "AGILE"},
    {"name": "aldehyde_04", "smiles": "O=C(/C=C/CCCCCCC)OCCCCCC=O", "source": "AGILE"},
    {"name": "aldehyde_05", "smiles": "O=C(/C=C\\CCCCCCC)OCCCCCC=O", "source": "AGILE"},
    {"name": "aldehyde_06", "smiles": "O=C(CCCCCCCCCC)OCCCCCC=O", "source": "AGILE"},
    {"name": "aldehyde_07", "smiles": "O=C(CCCCCCCCC#C)OCCCCCC=O", "source": "AGILE"},
    {"name": "aldehyde_08", "smiles": "O=C(CCCCCCCCC=C)OCCCCCC=O", "source": "AGILE"},
    {"name": "aldehyde_09", "smiles": "CCCCCCCCCCCCCCCC(OCCCCCC=O)=O", "source": "AGILE"},
    {"name": "aldehyde_10", "smiles": "CCCCCCCCCCCCCCCCCC(OCCCCCC=O)=O", "source": "AGILE"},
    {"name": "aldehyde_11", "smiles": "CCCCCCCC/C=C\\CCCCCCCC(OCCCCCC=O)=O", "source": "AGILE"},
    {"name": "aldehyde_12", "smiles": "CCCCC/C=C\\C/C=C\\CCCCCCCC(OCCCCCC=O)=O", "source": "AGILE"},
]

AGILE_ISOCYANIDES: list[dict] = [
    {"name": "isocyanide_01", "smiles": "CCCCCCCCCCCC[N+]#[C-]", "source": "AGILE"},
    {"name": "isocyanide_02", "smiles": "CCCCCCCCCCCCCC[N+]#[C-]", "source": "AGILE"},
    {"name": "isocyanide_03", "smiles": "CCCCCCCCCCCCCCCC[N+]#[C-]", "source": "AGILE"},
    {"name": "isocyanide_04", "smiles": "CCCCCCCCCCCCCCCCCC[N+]#[C-]", "source": "AGILE"},
    {"name": "isocyanide_05", "smiles": "CCCCCCCC/C=C\\CCCCCCCC[N+]#[C-]", "source": "AGILE"},
]

# ═══════════════════════════════════════════════════════════
# Lipid Combination 构建块 (文献策展)
# ═══════════════════════════════════════════════════════════

IONIZABLE_LIPIDS: list[dict] = [
    {"name": "SM-102", "smiles": "CCCCCCCCCCCCCCCCCCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCCCCCCC)N(C)CC(O)CO", "source": "Moderna"},
    {"name": "ALC-0315", "smiles": "CCCCCCCCCCCCCCCCC1=CC=C(C[C@@H](O)[C@H](N)CO)C=C1", "source": "Pfizer"},
    {"name": "MC3", "smiles": "CCCCCCCC/C=C\\C/C=C\\CCCCCCCCOC(=O)CCCN(C)C", "source": "Onpattro"},
    {"name": "KC2", "smiles": "CCCCCCCC/C=C\\C/C=C\\CCCCCCCCOCCN(C)C", "source": "Literature"},
    {"name": "C12-200", "smiles": "CCCCCCCCCCOC(=O)C1CC(C(=O)OCCCCCCCCCC)(C(=O)OCCCCCCCCCC)C1", "source": "Literature"},
]

HELPER_LIPIDS: list[dict] = [
    {"name": "DSPC", "smiles": "CCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)OC(=O)CCCCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCCCCCCC", "source": "Standard"},
    {"name": "DOPE", "smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COP(=O)(O)OCC(N)O)OC(=O)CCCCCCCC/C=C\\CCCCCCCC", "source": "Standard"},
    {"name": "DPPC", "smiles": "CCCCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)OC(=O)CCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCCCCC", "source": "Standard"},
    {"name": "DMPC", "smiles": "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)OC(=O)CCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCC", "source": "Standard"},
    {"name": "DLPC", "smiles": "CCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)OC(=O)CCCCCCCCCCCC)OC(=O)CCCCCCCCCCCC", "source": "Standard"},
]

PEG_LIPIDS: list[dict] = [
    {"name": "PEG2000-DMG", "smiles": "CCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC(CO)OC", "source": "Standard"},
    {"name": "DSPE-PEG2000", "smiles": "CCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(O)CO)OC(=O)CCCCCCCCCCCCCCCCCCCCCCCC", "source": "Standard"},
]
