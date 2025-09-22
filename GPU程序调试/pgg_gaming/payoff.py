"""公共物品博弈收益计算模块。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List

import networkx as nx

from .accelerator import AcceleratorController

try:  # 可选依赖
    import torch
except ImportError:  # pragma: no cover - 未安装 PyTorch 时触发
    torch = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PublicGoodsPayoffCalculator:
    """计算公共物品博弈中每个参与者的收益。"""

    investment: float = 1.0
    accelerator: AcceleratorController | None = None

    def __post_init__(self) -> None:
        self._accelerator = self.accelerator or AcceleratorController()

    # ------------------------------------------------------------------
    def calculate(self, graph: nx.Graph, synergy_factor: float) -> None:
        """根据协同系数计算网络内所有节点的收益。"""

        if graph.number_of_nodes() == 0:
            return

        _reset_profit(graph)

        if self._accelerator.is_enabled:
            try:
                self._calculate_gpu(graph, synergy_factor)
                return
            except Exception:  # pragma: no cover - 依赖外部硬件
                _LOGGER.exception("GPU 收益计算失败，自动退回 CPU 版本。")

        self._calculate_cpu(graph, synergy_factor)

    # ------------------------------------------------------------------
    def _calculate_cpu(self, graph: nx.Graph, synergy_factor: float) -> None:
        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            self._play_group_game(graph, node, neighbours, synergy_factor)

    def _play_group_game(
        self,
        graph: nx.Graph,
        host: int,
        neighbours: List[int],
        synergy_factor: float,
    ) -> None:
        group = [host, *neighbours]
        if not group:
            return

        cooperators = sum(_get_strategy(graph, member) for member in group)
        group_size = len(group)
        total_contribution = self.investment * cooperators
        total_return = synergy_factor * total_contribution
        average_return = total_return / group_size if group_size else 0.0

        for member in group:
            strategy = _get_strategy(graph, member)
            delta = average_return - self.investment if strategy else average_return
            graph.nodes[member]["profit"] += float(delta)

    # ------------------------------------------------------------------
    def _calculate_gpu(self, graph: nx.Graph, synergy_factor: float) -> None:
        if torch is None:
            raise RuntimeError("PyTorch 未安装，无法使用 GPU 计算。")

        device = torch.device(self._accelerator.device)
        nodes = list(graph.nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        strategies = torch.tensor(
            [float(_get_strategy(graph, node)) for node in nodes],
            dtype=torch.float64,
            device=device,
        )

        edges = list(graph.edges)
        if edges:
            row_idx: List[int] = []
            col_idx: List[int] = []
            for u, v in edges:
                ui = node_index[u]
                vi = node_index[v]
                row_idx.extend([ui, vi])
                col_idx.extend([vi, ui])

            indices = torch.tensor([row_idx, col_idx], dtype=torch.long, device=device)
            values = torch.ones(len(row_idx), dtype=torch.float64, device=device)
            adjacency = torch.sparse_coo_tensor(indices, values, (len(nodes), len(nodes)))
            adjacency = adjacency.coalesce()
        else:
            adjacency = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float64, device=device),
                (len(nodes), len(nodes)),
            )

        ones = torch.ones((len(nodes), 1), dtype=torch.float64, device=device)
        degree = torch.sparse.mm(adjacency, ones).squeeze(1)
        coop_neighbours = torch.sparse.mm(adjacency, strategies.unsqueeze(1)).squeeze(1)
        coop_count = coop_neighbours + strategies
        group_size = torch.clamp(degree + 1.0, min=1.0)

        r_tensor = torch.tensor(float(synergy_factor), dtype=torch.float64, device=device)
        investment_tensor = torch.tensor(float(self.investment), dtype=torch.float64, device=device)

        profit_avg = r_tensor * investment_tensor * coop_count / group_size
        base_contrib = profit_avg - investment_tensor * strategies
        neighbour_contrib = torch.sparse.mm(adjacency, profit_avg.unsqueeze(1)).squeeze(1)
        neighbour_contrib -= degree * investment_tensor * strategies
        total_profit = base_contrib + neighbour_contrib

        profits = total_profit.detach().cpu().numpy()
        for node, value in zip(nodes, profits):
            graph.nodes[node]["profit"] = float(value)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def _reset_profit(graph: nx.Graph) -> None:
    for node in graph.nodes:
        graph.nodes[node]["profit"] = 0.0


def _get_strategy(graph: nx.Graph, node: int) -> int:
    strategy = graph.nodes[node].get("strategy")
    if strategy is None:
        strategy = graph.nodes[node].get("select", 0)
    return int(strategy)


__all__ = ["PublicGoodsPayoffCalculator"]
