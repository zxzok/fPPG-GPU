"""策略初始化与更新模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, Iterable, List

import networkx as nx


@dataclass(slots=True)
class StrategyInitializer:
    """初始化网络节点的策略、收益与偏好。"""

    cooperative_probability: float = 0.5
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def apply(self, graph: nx.Graph) -> None:
        for node in graph.nodes:
            strategy = 1 if self._rng.random() < self.cooperative_probability else 0
            graph.nodes[node]["strategy"] = strategy
            graph.nodes[node]["select"] = strategy
            graph.nodes[node]["profit"] = 0.0
            graph.nodes[node].setdefault("preference", self._rng.random())


@dataclass(slots=True)
class StrategyUpdater:
    """根据收益情况调整节点策略。"""

    temperature: float = 0.5
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    def update(self, graph: nx.Graph) -> None:
        """执行标准的费米学习规则。"""

        snapshot = {node: graph.nodes[node]["strategy"] for node in graph.nodes}
        profits = {node: graph.nodes[node].get("profit", 0.0) for node in graph.nodes}
        updated = snapshot.copy()

        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            if not neighbours:
                continue

            target = self._rng.choice(neighbours)
            probability = _fermi_probability(profits[node], profits[target], self.temperature)
            _write_edge_probability(graph, node, target, probability)

            if self._rng.random() < probability:
                updated[node] = snapshot[target]

        _apply_strategy(graph, updated)

    def update_with_policy(self, graph: nx.Graph, alpha_g: float, alpha_i: float) -> None:
        """在标准学习规则上叠加政策导向影响。"""

        snapshot = {node: graph.nodes[node]["strategy"] for node in graph.nodes}
        profits = {node: graph.nodes[node].get("profit", 0.0) for node in graph.nodes}
        updated = snapshot.copy()

        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            if not neighbours:
                continue

            target = self._rng.choice(neighbours)
            base = _fermi_probability(profits[node], profits[target], self.temperature)
            preference = graph.nodes[node].get("preference", 0.0)
            policy = alpha_g * preference
            policy *= 1 if snapshot[target] == 1 else -1
            probability = alpha_i * base + policy
            probability = min(max(probability, 0.0), 1.0)

            _write_edge_probability(graph, node, target, probability)

            if self._rng.random() < probability:
                updated[node] = snapshot[target]

        _apply_strategy(graph, updated)


# ----------------------------------------------------------------------
# 指标计算
# ----------------------------------------------------------------------

def cooperation_ratio(graph: nx.Graph) -> float:
    """统计网络中的合作者比例。"""

    if graph.number_of_nodes() == 0:
        return 0.0
    total = sum(graph.nodes[node].get("strategy", 0) for node in graph.nodes)
    return total / graph.number_of_nodes()


def average_profit(graph: nx.Graph) -> float:
    """计算节点平均收益。"""

    if graph.number_of_nodes() == 0:
        return 0.0
    total = sum(graph.nodes[node].get("profit", 0.0) for node in graph.nodes)
    return total / graph.number_of_nodes()


# ----------------------------------------------------------------------
# 内部辅助函数
# ----------------------------------------------------------------------

def _fermi_probability(node_profit: float, neighbour_profit: float, temperature: float) -> float:
    delta = neighbour_profit - node_profit
    if delta >= 100:
        return 1.0
    if delta <= -100:
        return 0.0
    temperature = max(temperature, 1e-6)
    return 1.0 / (1.0 + math.exp(-delta / temperature))


def _write_edge_probability(graph: nx.Graph, source: int, target: int, probability: float) -> None:
    data = graph.get_edge_data(source, target, default={})
    data.setdefault("study_probability", {})
    data["study_probability"][source] = probability
    graph.add_edge(source, target, **data)


def _apply_strategy(graph: nx.Graph, strategies: Dict[int, int]) -> None:
    for node, strategy in strategies.items():
        graph.nodes[node]["strategy"] = strategy
        graph.nodes[node]["select"] = strategy


__all__ = [
    "StrategyInitializer",
    "StrategyUpdater",
    "average_profit",
    "cooperation_ratio",
]
