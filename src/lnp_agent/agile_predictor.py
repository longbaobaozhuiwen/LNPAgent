"""AGILE GIN 深度学习预测器 — 基于 Graph Isomorphism Network 的分子性能预测。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree

from rdkit import Chem

logger = logging.getLogger(__name__)

# 模型常量
NUM_ATOM_TYPE = 119
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 5
NUM_BOND_DIRECTION = 3


class GINEConv(MessagePassing):
    """Graph Isomorphism Network with Edge features."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x, edge_index, edge_attr):
        # Add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Self-loop edge attributes: bond_type=4 (self-loop), direction=0
        loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        loop_attr[:, 0] = 4  # self-loop bond type
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        return self.mlp(out)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class AGILEModel(nn.Module):
    """AGILE GIN 模型 (5层, 300d emb, 512d feat)。"""

    def __init__(self, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean"):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        self.feat_lin = nn.Linear(emb_dim, feat_dim)
        self.pred_head = nn.Linear(feat_dim, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long)

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = global_mean_pool(h, batch)
        h = self.feat_lin(h)
        out = self.pred_head(h)
        return h, out


class AGILEPredictor:
    """AGILE GIN 推理接口。

    v1.5 新增: encode(), encode_single() — 提取 512d 图嵌入。
    """

    FEAT_DIM = 512  # feat_lin 输出维度

    def __init__(self, ckpt_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = AGILEModel(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"AGILE model loaded from {ckpt_path}")

    @staticmethod
    def smiles_to_graph(smiles: str) -> Data | None:
        """RDKit SMILES → PyG Data 对象。"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 键类型映射 (rdkit BondType → int)
        BOND_TYPE_MAP = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }

        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetChiralTag(),
            ])
        x = torch.tensor(atom_features, dtype=torch.long)

        # 键特征 (双向)
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
            bond_dir = min(int(bond.GetBondDir()), NUM_BOND_DIRECTION - 1)  # clamp to [0, 2]
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append([bond_type, bond_dir])
            edge_attr.append([bond_type, bond_dir])

        if not edge_index:
            edge_index = torch.zeros((0, 2), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def predict(self, smiles_list: list[str]) -> np.ndarray:
        """批量预测 SMILES 列表。"""
        predictions = []
        for smi in smiles_list:
            graph = self.smiles_to_graph(smi)
            if graph is None:
                predictions.append(float("nan"))
                continue
            graph = graph.to(self.device)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
            _, out = self.model(graph)
            predictions.append(float(out.cpu().item()))
        return np.array(predictions)

    def predict_single(self, smiles: str) -> float:
        """单个 SMILES 预测。"""
        result = self.predict([smiles])
        return float(result[0])

    @torch.no_grad()
    def encode(self, smiles_list: list[str]) -> np.ndarray:
        """批量提取 SMILES 的 512d 图嵌入。无效 SMILES → 零向量。

        Returns:
            np.ndarray, shape (N, 512), dtype float32
        """
        embeddings = []
        for smi in smiles_list:
            graph = self.smiles_to_graph(smi)
            if graph is None:
                embeddings.append(np.zeros(self.FEAT_DIM, dtype=np.float32))
                continue
            graph = graph.to(self.device)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
            h, _ = self.model(graph)  # h: (1, 512)
            embeddings.append(h.cpu().numpy().flatten().astype(np.float32))
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, smiles: str) -> np.ndarray:
        """单个 SMILES → 512d 嵌入。"""
        return self.encode([smiles])[0]
