"""v2.3 湿实验模拟工具: Oracle + 重训练 + 主动学习轨迹可视化。

v2.3 改进:
- VRAM 隔离: retrain 前后清理 GPU 显存
- 原子写入: wet-lab 结果使用 atomic_write_csv 防止部分写入
- 40 候选物: exploitation (20) + exploration (20) = 40 个候选物
- 轨迹可视化: 1×4 面板 (Composite + immune signal A + immune signal B + Transfection)

保留工具:
- RunWetLabExperiment: 物理启发的隐藏 Ground Truth 生成器
- RetrainLNPPredictors: 增量追加训练数据 + 全量重训练
- PlotActiveLearningTrajectory: 多轮迭代性能攀升曲线
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from lnp_agent.path_utils import atomic_write_csv, resolve_round_file
from lnp_agent.tools.base import BaseTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# 抑制 RDKit 警告
RDLogger.logger().setLevel(RDLogger.ERROR)

# ═══════════════════════════════════════════════════════════
# Morgan FP 辅助函数
# ═══════════════════════════════════════════════════════════

def _smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """将 SMILES 转为 Morgan fingerprint (bit vector)。"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# 物理启发 Oracle (隐藏 Ground Truth)
# ═══════════════════════════════════════════════════════════

class WetLabOracle:
    """高保真湿实验模拟器: 基于物理启发的非线性函数 + 实验噪声。

    设计原则:
    - 确定性 (seeded): 相同输入 = 相同输出
    - 物理启发: 编码已知的 SAR 规律
    - 非线性: 简单线性模型无法完美拟合
    - 噪声: 模拟真实生物实验变异性
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Oracle 权重: 预生成的 Morgan FP 权重矩阵
        self._weights = {
            "immune_signal_a": self.rng.randn(2048) * 0.08,
            "immune_signal_b": self.rng.randn(2048) * 0.06,
            "tx_log1p": self.rng.randn(2048) * 0.10,
        }
        # 比例项权重
        self._ratio_weights = {
            "immune_signal_a": {"ratio1": 0.015, "np_ratio": 0.8, "intercept": 1.5},
            "immune_signal_b": {"ratio1": -0.005, "np_ratio": -0.1, "intercept": 0.4},
            "tx_log1p": {"ratio1": 0.02, "np_ratio": 5.0, "intercept": 2.0},
        }
        # 噪声参数
        self._noise_params = {
            "immune_signal_a": {"gaussian": 0.15, "multiplicative": 0.10},
            "immune_signal_b": {"gaussian": 0.05, "multiplicative": 0.12},
            "tx_log1p": {"gaussian": 0.20, "multiplicative": 0.15},
        }

    def predict(self, smiles: str, ratio1: float = 50.0, np_ratio: float = 0.08,
                experiment_seed: int | None = None) -> dict[str, float]:
        """生成单个配方的 "真实" 实验读数。"""
        fp = _smiles_to_morgan_fp(smiles)

        results = {}
        for endpoint in ["immune_signal_a", "immune_signal_b", "tx_log1p"]:
            # 1. 结构贡献 (Morgan FP + 非线性激活)
            fp_score = fp @ self._weights[endpoint]

            # 2. 比例贡献
            rw = self._ratio_weights[endpoint]
            ratio_score = rw["ratio1"] * ratio1 + rw["np_ratio"] * np_ratio

            # 3. 物理启发的非线性变换
            total = fp_score + ratio_score
            if endpoint == "immune_signal_a":
                # 疏水性越大炎症越强 (指数增长)
                raw = np.exp(0.25 * np.clip(total, -3, 3)) * 0.8 + 0.5
            elif endpoint == "immune_signal_b":
                # 对头部电荷和 PEG 链敏感 (sigmoid)
                raw = 1.0 / (1.0 + np.exp(-total)) * 1.4 + 0.05
            else:  # tx_log1p
                # 分子量最优区间呈钟形关系
                raw = np.maximum(0.1, -((total - 1.5) ** 2) + 4.5) + 1.0

            # 4. 添加实验噪声
            noise_rng = np.random.RandomState(experiment_seed or 0)
            params = self._noise_params[endpoint]
            noisy = raw * (1 + noise_rng.normal(0, params["multiplicative"]))
            noisy += noise_rng.normal(0, params["gaussian"])
            noisy = max(0.0, float(noisy))

            results[endpoint] = round(noisy, 4)

        return results


# ═══════════════════════════════════════════════════════════
# Tool 1: RunWetLabExperiment
# ═══════════════════════════════════════════════════════════

class RunWetLabExperiment(BaseTool):
    """模拟湿实验: 为 40 个候选物生成高保真实验读数。"""

    def __init__(self, data_manager):
        self.dm = data_manager
        self._oracle = WetLabOracle(seed=42)
        self.definition = ToolDefinition(
            name="run_wet_lab_experiment",
            description=(
                "Run simulated wet-lab experiment on selected candidates. "
                "Takes exploitation (20) and exploration (20) candidates, "
                "generates realistic immune signal A, immune signal B, and transfection readings "
                "with experimental noise. Saves results to wet_lab_results_round_N.csv."
            ),
            parameters={
                "exploitation_path": {
                    "type": "string",
                    "description": "Path to exploitation candidates CSV (20 rows).",
                },
                "exploration_path": {
                    "type": "string",
                    "description": "Path to exploration candidates CSV (20 rows).",
                },
                "round_number": {
                    "type": "integer",
                    "description": "Current active learning round number (1, 2, or 3).",
                },
            },
            required=["exploitation_path", "exploration_path", "round_number"],
        )

    def execute(
        self,
        exploitation_path: str | None = None,
        exploration_path: str | None = None,
        round_number: int = 1,
        **kwargs,
    ) -> ToolResult:
        from lnp_agent.path_utils import sanitize_output_path

        # 1. 路由输入路径 (v2.3: 使用 resolve_round_file 处理 LLM 编造的文件名)
        if exploitation_path:
            exploitation_path = str(resolve_round_file(
                exploitation_path,
                ["exploitation_round_*.csv", "*exploitation*.csv", "*exploit*.csv"]))
        if exploration_path:
            exploration_path = str(resolve_round_file(
                exploration_path,
                ["exploration_round_*.csv", "*exploration*.csv", "*explor*.csv"]))

        # 2. 加载候选物
        try:
            exp_df = pd.read_csv(exploitation_path) if exploitation_path else pd.DataFrame()
            expl_df = pd.read_csv(exploration_path) if exploration_path else pd.DataFrame()
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to load candidates: {e}")

        if len(exp_df) == 0 and len(expl_df) == 0:
            return ToolResult(success=False, output="",
                              error="No candidates loaded. Provide both paths.")

        # 3. 标记候选类型
        exp_df["candidate_type"] = "exploitation"
        expl_df["candidate_type"] = "exploration"

        # 4. 合并
        all_df = pd.concat([exp_df, expl_df], ignore_index=True)
        all_df["round"] = round_number
        all_df["measurement_type"] = "synthetic_oracle"
        all_df["measurement_provenance"] = "LNPAgent WetLabOracle; not a physical experiment"
        all_df["experiment_id"] = [
            f"R{round_number}_{i+1:03d}" for i in range(len(all_df))
        ]

        # 5. 获取 SMILES (优先 lipid1_smiles, 备选 combined_smiles)
        smiles_col = "lipid1_smiles" if "lipid1_smiles" in all_df.columns else "combined_smiles"
        if smiles_col not in all_df.columns:
            return ToolResult(success=False, output="",
                              error=f"No SMILES column found (tried lipid1_smiles, combined_smiles)")

        # 6. Oracle 预测 + 噪声
        ratio1_col = "ratio1" if "ratio1" in all_df.columns else None
        np_ratio_col = "np_ratio" if "np_ratio" in all_df.columns else None

        immune_signal_a_vals, immune_signal_b_vals, tx_vals = [], [], []
        for idx, row in all_df.iterrows():
            smiles = str(row[smiles_col])
            ratio1 = float(row[ratio1_col]) if ratio1_col and pd.notna(row.get("ratio1")) else 50.0
            np_ratio = float(row[np_ratio_col]) if np_ratio_col and pd.notna(row.get("np_ratio")) else 0.08

            pred = self._oracle.predict(
                smiles, ratio1=ratio1, np_ratio=np_ratio,
                experiment_seed=42 + round_number * 1000 + idx,
            )
            immune_signal_a_vals.append(pred["immune_signal_a"])
            immune_signal_b_vals.append(pred["immune_signal_b"])
            tx_vals.append(pred["tx_log1p"])

        all_df["measured_immune_signal_a"] = immune_signal_a_vals
        all_df["measured_immune_signal_b"] = immune_signal_b_vals
        all_df["measured_tx_log1p"] = tx_vals

        # 7. 保存结果 (atomic write)
        out = sanitize_output_path(
            None, f"wet_lab_results_round_{round_number}.csv",
            subdir="wet_lab",
        )
        atomic_write_csv(all_df, out)

        # 8. 统计摘要
        summary = {
            "round": round_number,
            "total_candidates": len(all_df),
            "exploitation_count": len(exp_df),
            "exploration_count": len(expl_df),
            "immune_signal_a_mean": round(float(all_df["measured_immune_signal_a"].mean()), 4),
            "immune_signal_a_range": f"{all_df['measured_immune_signal_a'].min():.4f} - {all_df['measured_immune_signal_a'].max():.4f}",
            "immune_signal_b_mean": round(float(all_df["measured_immune_signal_b"].mean()), 4),
            "immune_signal_b_range": f"{all_df['measured_immune_signal_b'].min():.4f} - {all_df['measured_immune_signal_b'].max():.4f}",
            "tx_mean": round(float(all_df["measured_tx_log1p"].mean()), 4),
            "tx_range": f"{all_df['measured_tx_log1p'].min():.4f} - {all_df['measured_tx_log1p'].max():.4f}",
            "output_file": str(out),
        }

        logger.info(f"Wet-lab results saved: {out} ({len(all_df)} rows)")
        return ToolResult(success=True, output=json.dumps(summary, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════
# Tool 2: RetrainLNPPredictors
# ═══════════════════════════════════════════════════════════

class RetrainLNPPredictors(BaseTool):
    """重训练预测模型: 将湿实验结果追加到训练集并重新训练。"""

    def __init__(self, data_manager):
        self.dm = data_manager
        self.definition = ToolDefinition(
            name="retrain_lnp_predictors",
            description=(
                "Retrain all LNP prediction models with new wet-lab experimental data. "
                "Appends wet-lab results to the training set, clears model cache, "
                "and retrains Ridge/HGBR/Huber predictors for all endpoints. "
                "After retraining, predictions should be more accurate for the "
                "explored chemical space."
            ),
            parameters={
                "wet_lab_results_path": {
                    "type": "string",
                    "description": "Path to wet_lab_results_round_N.csv.",
                },
            },
            required=["wet_lab_results_path"],
        )

    def execute(
        self,
        wet_lab_results_path: str | None = None,
        **kwargs,
    ) -> ToolResult:
        # 1. 路由输入路径 (v2.3: 使用 resolve_round_file 处理模糊路径)
        if wet_lab_results_path:
            wet_lab_results_path = str(resolve_round_file(
                wet_lab_results_path,
                ["wet_lab_results_round_*.csv", "*wet_lab*.csv"],
                subdir="wet_lab"))

        if not wet_lab_results_path or not Path(wet_lab_results_path).exists():
            return ToolResult(success=False, output="",
                              error=f"Wet-lab results not found: {wet_lab_results_path}")

        # 2. 加载湿实验结果
        try:
            wet_lab_df = pd.read_csv(wet_lab_results_path)
        except Exception as e:
            return ToolResult(success=False, output="",
                              error=f"Failed to load wet-lab results: {e}")

        if len(wet_lab_df) == 0:
            return ToolResult(success=False, output="",
                              error="Wet-lab results file is empty")

        # 3. 检查 DataManager 是否为 DataManagerV2
        from lnp_agent.data_manager_v2 import DataManagerV2
        if not isinstance(self.dm, DataManagerV2):
            return ToolResult(success=False, output="",
                              error="DataManager is not DataManagerV2 (required for retraining)")

        # 4. 追加数据并重训练 (with VRAM cleanup)
        from lnp_agent.gpu_utils import clear_gpu_memory

        try:
            clear_gpu_memory("before retrain_lnp_predictors")
            old_size = len(self.dm.df)
            new_size = self.dm.append_wet_lab_results(wet_lab_df)
            retrain_results = self.dm.force_retrain_all()
            clear_gpu_memory("after retrain_lnp_predictors")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return ToolResult(success=False, output="",
                              error=f"Retraining error: {type(e).__name__}: {e}")

        # 5. 报告
        success_count = sum(1 for v in retrain_results.values() if v.get("status") == "success")
        error_count = sum(1 for v in retrain_results.values() if v.get("status") == "error")

        summary = {
            "dataset_size_before": old_size,
            "dataset_size_after": new_size,
            "new_rows_added": len(wet_lab_df),
            "models_retrained": success_count,
            "model_errors": error_count,
            "total_configs": len(retrain_results),
        }

        logger.info(f"Retraining complete: {new_size} rows, {success_count}/{len(retrain_results)} models")
        return ToolResult(success=True, output=json.dumps(summary, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════
# Tool 3: PlotActiveLearningTrajectory
# ═══════════════════════════════════════════════════════════

class PlotActiveLearningTrajectory(BaseTool):
    """绘制主动学习轨迹图: 展示候选物得分随轮次的改善。"""

    def __init__(self, data_manager=None):
        self.dm = data_manager
        self.definition = ToolDefinition(
            name="plot_active_learning_trajectory",
            description=(
                "Plot active learning trajectory showing how candidate "
                "experimental scores improve across rounds. "
                "Reads all wet_lab_results_round_N.csv files automatically."
            ),
            parameters={
                "save_path": {
                    "type": "string",
                    "description": "Output PNG path (default: auto in Results/figures/).",
                },
            },
            required=[],
        )

    def execute(self, save_path: str | None = None, **kwargs) -> ToolResult:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from lnp_agent.path_utils import sanitize_save_path
        from lnp_agent.paths import WET_LAB_DIR

        # 1. 加载所有轮次数据
        if not WET_LAB_DIR.exists():
            return ToolResult(success=False, output="",
                              error=f"Wet-lab directory not found: {WET_LAB_DIR}")

        round_data = {}
        for csv_path in sorted(WET_LAB_DIR.glob("wet_lab_results_round_*.csv")):
            try:
                round_num = int(csv_path.stem.split("_")[-1])
                df = pd.read_csv(csv_path)
                round_data[round_num] = df
            except (ValueError, Exception) as e:
                logger.warning(f"Skipping {csv_path}: {e}")
                continue

        if not round_data:
            return ToolResult(success=False, output="",
                              error="No wet-lab result files found")

        # 2. 合并所有轮次数据用于全局归一化
        all_data = pd.concat(round_data.values(), ignore_index=True)

        def normalize_series(s):
            mn, mx = s.min(), s.max()
            if mx == mn:
                return pd.Series(0.5, index=s.index)
            return (s - mn) / (mx - mn)

        # 全局归一化
        immune_signal_a_norm_global = normalize_series(all_data["measured_immune_signal_a"])
        immune_signal_b_norm_global = normalize_series(all_data["measured_immune_signal_b"])
        tx_norm_global = normalize_series(all_data["measured_tx_log1p"])

        # 3. 计算每轮综合得分
        all_data["immune_signal_a_norm"] = immune_signal_a_norm_global
        all_data["immune_signal_b_norm"] = immune_signal_b_norm_global
        all_data["tx_norm"] = tx_norm_global
        all_data["score"] = -all_data["immune_signal_a_norm"] - all_data["immune_signal_b_norm"] + all_data["tx_norm"]

        round_scores = {}
        for rnd in sorted(round_data.keys()):
            rnd_data = all_data[all_data["round"] == rnd]
            round_scores[rnd] = {
                "mean": float(rnd_data["score"].mean()),
                "std": float(rnd_data["score"].std()),
                "exploitation_mean": float(
                    rnd_data[rnd_data["candidate_type"] == "exploitation"]["score"].mean()
                ) if "exploitation" in rnd_data["candidate_type"].values else 0.0,
                "exploration_mean": float(
                    rnd_data[rnd_data["candidate_type"] == "exploration"]["score"].mean()
                ) if "exploration" in rnd_data["candidate_type"].values else 0.0,
            }

        # 4. 计算每轮端点均值 (用于分面板)
        round_endpoints = {}
        for rnd in sorted(round_data.keys()):
            rnd_data = all_data[all_data["round"] == rnd]
            round_endpoints[rnd] = {
                "immune_signal_a_mean": float(rnd_data["measured_immune_signal_a"].mean()),
                "immune_signal_a_std": float(rnd_data["measured_immune_signal_a"].std()),
                "immune_signal_b_mean": float(rnd_data["measured_immune_signal_b"].mean()),
                "immune_signal_b_std": float(rnd_data["measured_immune_signal_b"].std()),
                "tx_mean": float(rnd_data["measured_tx_log1p"].mean()),
                "tx_std": float(rnd_data["measured_tx_log1p"].std()),
                "n": len(rnd_data),
            }

        # 5. 绘图 (Nature 风格, 1x4 面板)
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.linewidth': 0.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'legend.fontsize': 6,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })

        fig, axes = plt.subplots(1, 4, figsize=(14, 3))

        rounds = sorted(round_scores.keys())
        all_means = [round_scores[r]["mean"] for r in rounds]
        all_stds = [round_scores[r]["std"] for r in rounds]
        exp_means = [round_scores[r]["exploitation_mean"] for r in rounds]
        expl_means = [round_scores[r]["exploration_mean"] for r in rounds]

        # --- Panel 0: Composite Score ---
        ax = axes[0]
        ax.plot(rounds, all_means, 'o-', color='#2196F3', linewidth=1.5,
                markersize=6, label='All candidates', zorder=3)
        ax.fill_between(rounds,
                        [m - s for m, s in zip(all_means, all_stds)],
                        [m + s for m, s in zip(all_means, all_stds)],
                        alpha=0.2, color='#2196F3')

        # Exploitation 子线
        ax.plot(rounds, exp_means, 's--', color='#EA4335', linewidth=1.0,
                markersize=4, label='Exploitation', zorder=2, alpha=0.8)

        # Exploration 子线
        ax.plot(rounds, expl_means, 'D--', color='#4285F4', linewidth=1.0,
                markersize=4, label='Exploration', zorder=2, alpha=0.8)

        # n=N 候选物数量注释
        for i, rnd in enumerate(rounds):
            n = round_endpoints[rnd]["n"]
            ax.annotate(f'n={n}', (rnd, all_means[i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=5, ha='center', color='#555555')

        ax.set_xlabel('Active Learning Round')
        ax.set_ylabel('Experimental Score')
        ax.set_xticks(rounds)
        ax.legend(frameon=False, fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- Panel 1: immune signal A trajectory ---
        ax1 = axes[1]
        immune_signal_a_means = [round_endpoints[r]["immune_signal_a_mean"] for r in rounds]
        immune_signal_a_stds = [round_endpoints[r]["immune_signal_a_std"] for r in rounds]
        ax1.plot(rounds, immune_signal_a_means, 'o-', color='#EA4335', linewidth=1.5,
                 markersize=6, label='immune signal A')
        ax1.fill_between(rounds,
                         [m - s for m, s in zip(immune_signal_a_means, immune_signal_a_stds)],
                         [m + s for m, s in zip(immune_signal_a_means, immune_signal_a_stds)],
                         alpha=0.2, color='#EA4335')
        ax1.set_xlabel('Active Learning Round')
        ax1.set_ylabel('immune signal A (pg/mL)')
        ax1.set_xticks(rounds)
        ax1.legend(frameon=False, fontsize=6)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # --- Panel 2: immune signal B trajectory ---
        ax2 = axes[2]
        immune_signal_b_means = [round_endpoints[r]["immune_signal_b_mean"] for r in rounds]
        immune_signal_b_stds = [round_endpoints[r]["immune_signal_b_std"] for r in rounds]
        ax2.plot(rounds, immune_signal_b_means, 'o-', color='#34A853', linewidth=1.5,
                 markersize=6, label='immune signal B')
        ax2.fill_between(rounds,
                         [m - s for m, s in zip(immune_signal_b_means, immune_signal_b_stds)],
                         [m + s for m, s in zip(immune_signal_b_means, immune_signal_b_stds)],
                         alpha=0.2, color='#34A853')
        ax2.set_xlabel('Active Learning Round')
        ax2.set_ylabel('immune signal B (pg/mL)')
        ax2.set_xticks(rounds)
        ax2.legend(frameon=False, fontsize=6)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # --- Panel 3: Transfection trajectory ---
        ax3 = axes[3]
        tx_means = [round_endpoints[r]["tx_mean"] for r in rounds]
        tx_stds = [round_endpoints[r]["tx_std"] for r in rounds]
        ax3.plot(rounds, tx_means, 'o-', color='#4285F4', linewidth=1.5,
                 markersize=6, label='Transfection')
        ax3.fill_between(rounds,
                         [m - s for m, s in zip(tx_means, tx_stds)],
                         [m + s for m, s in zip(tx_means, tx_stds)],
                         alpha=0.2, color='#4285F4')
        ax3.set_xlabel('Active Learning Round')
        ax3.set_ylabel('Transfection (log1p)')
        ax3.set_xticks(rounds)
        ax3.legend(frameon=False, fontsize=6)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # 6. 保存
        fig.tight_layout(pad=0.5)
        out = sanitize_save_path(save_path, "active_learning_trajectory.png")
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

        summary = {
            "rounds": rounds,
            "scores": {str(r): round_scores[r] for r in rounds},
            "output_file": str(out),
        }
        logger.info(f"Active learning trajectory saved: {out}")
        return ToolResult(success=True, output=json.dumps(summary, ensure_ascii=False))
