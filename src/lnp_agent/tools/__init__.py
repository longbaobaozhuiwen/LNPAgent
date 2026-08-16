"""Tool registration and initialization."""
from __future__ import annotations

from lnp_agent.sandbox import Sandbox
from lnp_agent.tools.base import BaseTool
from lnp_agent.tools.standard import (
    BashExecutor,
    EditTool,
    GlobTool,
    GrepTool,
    ReadTool,
    WriteTool,
)


def create_standard_tools(sandbox: Sandbox) -> dict[str, BaseTool]:
    """创建并注册所有标准工具。"""
    return {
        "bash_executor": BashExecutor(sandbox),
        "glob": GlobTool(sandbox),
        "grep": GrepTool(sandbox),
        "read": ReadTool(sandbox),
        "write": WriteTool(sandbox),
        "edit": EditTool(sandbox),
    }


def create_domain_tools(data_manager) -> dict[str, BaseTool]:
    """创建领域工具。"""
    from lnp_agent.tools.domain import (
        CheckCompliance,
        EnumerateMissingCells,
        PredictLNPPerformance,
        QueryParetoFront,
    )
    return {
        "check_compliance": CheckCompliance(),
        "predict_lnp_performance": PredictLNPPerformance(data_manager),
        "query_pareto_front": QueryParetoFront(data_manager),
        "enumerate_missing_cells": EnumerateMissingCells(data_manager),
    }


def create_active_learning_tools(data_manager) -> dict[str, BaseTool]:
    """创建主动学习工具 (v2.3: 含湿实验 + 重训练 + 原子写入)。"""
    from lnp_agent.tools.active_learning import (
        BatchPredictLNP,
        FilterExploitationBatch,
        FilterExplorationBatch,
    )
    from lnp_agent.tools.wet_lab import (
        RunWetLabExperiment,
        RetrainLNPPredictors,
        PlotActiveLearningTrajectory,
    )
    return {
        "batch_predict_lnp": BatchPredictLNP(data_manager),
        "filter_exploitation_batch": FilterExploitationBatch(data_manager),
        "filter_exploration_batch": FilterExplorationBatch(data_manager),
        "run_wet_lab_experiment": RunWetLabExperiment(data_manager),
        "retrain_lnp_predictors": RetrainLNPPredictors(data_manager),
        "plot_active_learning_trajectory": PlotActiveLearningTrajectory(data_manager),
    }


def create_library_tools(data_manager=None) -> dict[str, BaseTool]:
    """创建虚拟库生成工具。"""
    from lnp_agent.tools.generate_library import GenerateVirtualLibrary
    return {
        "generate_virtual_library": GenerateVirtualLibrary(data_manager),
    }


def create_visualization_tools() -> dict[str, BaseTool]:
    """创建可视化工具。"""
    from lnp_agent.tools.visualization import PlotParetoFront, PlotChemicalSpaceUMAP
    return {
        "plot_pareto_front": PlotParetoFront(),
        "plot_chemical_space_umap": PlotChemicalSpaceUMAP(),
    }


def create_external_tools() -> dict[str, BaseTool]:
    """Create adapters for locally installed third-party LNP tools."""
    from lnp_agent.tools.external import GenerateLaMGenMolecules, RunCOMETInference

    return {
        "run_comet_inference": RunCOMETInference(),
        "generate_lamgen_molecules": GenerateLaMGenMolecules(),
    }


def create_all_tools(sandbox: Sandbox, data_manager=None) -> dict[str, BaseTool]:
    """创建所有工具 (v2.3: 19 工具 = 6 标准 + 4 领域 + 6 AL + 1 库 + 2 可视化)。"""
    tools = create_standard_tools(sandbox)
    if data_manager is not None:
        tools.update(create_domain_tools(data_manager))
        tools.update(create_active_learning_tools(data_manager))
        tools.update(create_library_tools(data_manager))
    tools.update(create_visualization_tools())
    tools.update(create_external_tools())
    return tools
