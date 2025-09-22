"""网络生成与规模平滑模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable, List

import networkx as nx

from .config import NetworkConfig


@dataclass(slots=True)
class NetworkFactory:
    """根据配置生成不同拓扑类型的网络。"""

    seed: int | None = None
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:  # 初始化随机数发生器
        self._random = random.Random(self.seed)

    # ------------------------------------------------------------------
    def create(self, config: NetworkConfig) -> nx.Graph:
        """创建一个符合配置的图。"""

        generator_seed = config.seed if config.seed is not None else self.seed

        if config.topology == "ba":
            m = max(1, round(config.mean_degree / 2))
            graph = nx.barabasi_albert_graph(config.size, m, seed=generator_seed)
        elif config.topology == "ws":
            k = max(2, config.mean_degree if config.mean_degree % 2 == 0 else config.mean_degree + 1)
            graph = nx.watts_strogatz_graph(
                config.size,
                k,
                config.ws_rewire_prob,
                seed=generator_seed,
            )
        elif config.topology in {"er", "random"}:
            if config.er_edge_prob is not None:
                p = config.er_edge_prob
            else:
                p = min(1.0, max(0.0, config.mean_degree / max(config.size - 1, 1)))
            graph = nx.erdos_renyi_graph(config.size, p, seed=generator_seed)
        elif config.topology == "regular":
            graph = nx.random_regular_graph(config.mean_degree, config.size, seed=generator_seed)
        elif config.topology == "tree":
            graph = nx.random_tree(config.size, seed=generator_seed)
        else:
            msg = f"暂不支持的网络类型：{config.topology}"
            raise ValueError(msg)

        initialise_node_attributes(graph, self._random)
        return graph


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------

def initialise_node_attributes(graph: nx.Graph, rng: random.Random) -> None:
    """为网络节点设置默认属性。"""

    for node in graph.nodes:
        graph.nodes[node]["strategy"] = 1 if rng.random() < 0.5 else 0
        graph.nodes[node]["select"] = graph.nodes[node]["strategy"]
        graph.nodes[node]["profit"] = 0.0
        graph.nodes[node]["preference"] = rng.random()


def smooth_size_sequence(sizes: Iterable[int], max_scale: float = 0.5) -> List[int]:
    """对网络规模序列进行平滑处理，避免相邻规模变化过大。"""

    sizes = list(sizes)
    if len(sizes) < 2:
        return sizes

    smoothed = [sizes[0]]
    for current, target in zip(sizes, sizes[1:]):
        lower = current * (1 - max_scale)
        upper = current * (1 + max_scale)

        if lower <= target <= upper:
            smoothed.append(target)
            continue

        step_ratio = 1 + max_scale if target > current else 1 - max_scale
        steps = int(math.log(target / current, step_ratio)) if current != target else 0
        steps = max(steps, 1)

        value = current
        for _ in range(steps):
            value *= step_ratio
            smoothed.append(int(round(value)))

        if smoothed[-1] != target:
            smoothed.append(target)

    return smoothed


__all__ = ["NetworkFactory", "smooth_size_sequence", "initialise_node_attributes"]
