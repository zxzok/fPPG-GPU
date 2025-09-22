"""模拟参数配置模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence


NetworkType = Literal["ba", "ws", "er", "regular", "tree", "random"]


@dataclass(slots=True)
class NetworkConfig:
    """描述网络生成所需的关键参数。"""

    topology: NetworkType
    size: int
    mean_degree: int = 4
    ws_rewire_prob: float = 0.1
    er_edge_prob: Optional[float] = None
    seed: Optional[int] = None


@dataclass(slots=True)
class SimulationConfig:
    """公共物品博弈仿真配置。"""

    network: NetworkConfig
    rounds: int = 200
    synergy_factor: float = 3.0
    cooperative_investment: float = 1.0
    learning_temperature: float = 0.5
    policy_alpha_g: float = 0.0
    policy_alpha_i: float = 1.0
    initial_cooperation: float = 0.5
    enable_policy_bias: bool = False
    random_seed: Optional[int] = None


@dataclass(slots=True)
class BatchSimulationConfig:
    """批量仿真任务配置。"""

    size_sequence: Sequence[int] = field(default_factory=list)
    max_scale: float = 0.5


__all__ = [
    "BatchSimulationConfig",
    "NetworkConfig",
    "NetworkType",
    "SimulationConfig",
]
