"""虚拟库生成工具 (v2.0 扩展版)。

v2.0 变更:
- 扩展默认构建块 (从 building_blocks.py 加载)
- Ugi-3CR 无参数时自动使用 AGILE 默认构建块
- 保留 _raw fallback 处理 (来自 v1.9)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from lnp_agent.tools.base import BaseTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class GenerateVirtualLibrary(BaseTool):
    """生成虚拟候选配方库。

    支持两种模式:
    - ugi3cr: Ugi 三组分反应 (胺 + 醛 + 异腈 → 产物)
    - lipid_combination: 脂质角色组合 (ionizable + helper + PEG)
    """

    def __init__(self, data_manager=None):
        self.data_manager = data_manager
        self.definition = ToolDefinition(
            name="generate_virtual_library",
            description=(
                "Generate a virtual library of candidate LNP formulations. "
                "Use 'ugi3cr' mode for Ugi three-component reaction "
                "(provide a_smiles, b_smiles, c_smiles as comma-separated SMILES), "
                "or 'lipid_combination' mode for lipid role combinations. "
                "If no SMILES provided, defaults to expanded building block libraries."
            ),
            parameters={
                "mode": {
                    "type": "string",
                    "description": (
                        "Library generation mode: 'ugi3cr' for Ugi three-component reaction, "
                        "'lipid_combination' for lipid role combination."
                    ),
                    "enum": ["ugi3cr", "lipid_combination"],
                },
                "a_smiles": {
                    "type": "string",
                    "description": (
                        "Comma-separated SMILES for component A. "
                        "ugi3cr: amines. lipid_combination: ionizable lipids (name:smiles pairs)."
                    ),
                },
                "b_smiles": {
                    "type": "string",
                    "description": (
                        "Comma-separated SMILES for component B. "
                        "ugi3cr: aldehydes. lipid_combination: helper lipids (name:smiles pairs)."
                    ),
                },
                "c_smiles": {
                    "type": "string",
                    "description": (
                        "Comma-separated SMILES for component C. "
                        "ugi3cr: isocyanides. lipid_combination: PEG lipids (name:smiles pairs)."
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save the generated library CSV. "
                                   "Defaults to working_data/virtual_library.csv.",
                },
            },
            required=[],
        )

    def execute(self, **kwargs) -> ToolResult:
        # 处理 _raw fallback (Gemma 4 解析伪影)
        if "_raw" in kwargs and len(kwargs) == 1:
            raw = kwargs["_raw"]
            import re
            mode_match = re.search(r'mode:\s*["\']?(\w+)["\']?', raw)
            if mode_match:
                kwargs = {"mode": mode_match.group(1)}
                path_match = re.search(r'output_path:\s*["\']?([^"\',}]+)["\']?', raw)
                if path_match:
                    kwargs["output_path"] = path_match.group(1).strip()
            else:
                logger.warning(f"Could not parse _raw arguments, using lipid_combination default: {raw[:100]}")
                kwargs = {"mode": "lipid_combination"}

        mode = kwargs.get("mode", "lipid_combination")
        output_path = kwargs.get("output_path")

        try:
            if mode == "ugi3cr":
                return self._generate_ugi3cr(kwargs, output_path)
            elif mode == "lipid_combination":
                return self._generate_lipid_combination(kwargs, output_path)
            else:
                return ToolResult(
                    success=False, output="",
                    error=f"Unknown mode: {mode}. Use 'ugi3cr' or 'lipid_combination'.",
                )
        except Exception as e:
            logger.error(f"Library generation failed: {e}")
            return ToolResult(success=False, output="", error=str(e))

    def _generate_ugi3cr(self, kwargs: dict, output_path: str | None) -> ToolResult:
        """Ugi-3CR 虚拟库生成。"""
        from lnp_agent.cheminformatics import generate_ugi3cr_library
        from lnp_agent.paths import WORKING_DATA_DIR

        a_raw = kwargs.get("a_smiles", "")
        b_raw = kwargs.get("b_smiles", "")
        c_raw = kwargs.get("c_smiles", "")

        # v2.0: 无参数时使用 AGILE 默认构建块
        if not a_raw or not b_raw or not c_raw:
            from lnp_agent.building_blocks import AGILE_AMINES, AGILE_ALDEHYDES, AGILE_ISOCYANIDES
            if not a_raw:
                a_raw = ",".join(b["smiles"] for b in AGILE_AMINES)
                logger.info(f"Using {len(AGILE_AMINES)} default AGILE amines")
            if not b_raw:
                b_raw = ",".join(b["smiles"] for b in AGILE_ALDEHYDES)
                logger.info(f"Using {len(AGILE_ALDEHYDES)} default AGILE aldehydes")
            if not c_raw:
                c_raw = ",".join(b["smiles"] for b in AGILE_ISOCYANIDES)
                logger.info(f"Using {len(AGILE_ISOCYANIDES)} default AGILE isocyanides")

        a_list = [s.strip() for s in a_raw.split(",") if s.strip()]
        b_list = [s.strip() for s in b_raw.split(",") if s.strip()]
        c_list = [s.strip() for s in c_raw.split(",") if s.strip()]

        if not a_list or not b_list or not c_list:
            return ToolResult(
                success=False, output="",
                error="No valid SMILES found after parsing.",
            )

        result = generate_ugi3cr_library(a_list, b_list, c_list)

        out = Path(output_path) if output_path else WORKING_DATA_DIR / "virtual_library.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        result.library_df.to_csv(out, index=False)

        summary = json.dumps({
            "output_file": str(out),
            "n_total": result.n_total,
            "n_valid": result.n_valid,
            "n_invalid": result.n_invalid,
            "generation_mode": result.generation_mode,
        })
        logger.info(f"Ugi-3CR library: {result.n_valid}/{result.n_total} valid → {out}")
        return ToolResult(success=True, output=summary)

    def _generate_lipid_combination(self, kwargs: dict, output_path: str | None) -> ToolResult:
        """脂质角色组合库生成。"""
        from lnp_agent.cheminformatics import generate_lipid_combination_library
        from lnp_agent.paths import WORKING_DATA_DIR

        # 解析 name:smiles 对或纯 SMILES
        ionizable = self._parse_lipid_input(kwargs.get("a_smiles", ""), "ionizable")
        helper = self._parse_lipid_input(kwargs.get("b_smiles", ""), "helper")
        peg = self._parse_lipid_input(kwargs.get("c_smiles", ""), "peg")

        # v2.0: 从 building_blocks.py 加载扩展默认值
        if not ionizable or not helper or not peg:
            defaults = self._get_default_lipids()
            if not ionizable:
                ionizable = defaults.get("ionizable", [])
            if not helper:
                helper = defaults.get("helper", [])
            if not peg:
                peg = defaults.get("peg", [])

        if not ionizable or not helper or not peg:
            return ToolResult(
                success=False, output="",
                error="lipid_combination mode requires ionizable, helper, and PEG lipids.",
            )

        result = generate_lipid_combination_library(ionizable, helper, peg)

        out = Path(output_path) if output_path else WORKING_DATA_DIR / "virtual_library.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        result.library_df.to_csv(out, index=False)

        summary = json.dumps({
            "output_file": str(out),
            "n_total": result.n_total,
            "n_valid": result.n_valid,
            "n_invalid": result.n_invalid,
            "generation_mode": result.generation_mode,
        })
        logger.info(f"Lipid combination library: {result.n_total} formulations → {out}")
        return ToolResult(success=True, output=summary)

    def _parse_lipid_input(self, raw: str, role: str) -> list[dict]:
        """解析脂质输入 (name:smiles 对或纯 SMILES)。"""
        if not raw:
            return []
        entries = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                name, smiles = part.split(":", 1)
                entries.append({"name": name.strip(), "smiles": smiles.strip()})
            else:
                entries.append({"name": f"{role}_{len(entries)+1}", "smiles": part})
        return entries

    def _get_default_lipids(self) -> dict:
        """v2.0: 从 building_blocks.py 加载扩展默认脂质。"""
        from lnp_agent.building_blocks import IONIZABLE_LIPIDS, HELPER_LIPIDS, PEG_LIPIDS
        return {
            "ionizable": IONIZABLE_LIPIDS,
            "helper": HELPER_LIPIDS,
            "peg": PEG_LIPIDS,
        }
