"""pgg_gaming 模块提供公共物品博弈的核心能力。"""

from .accelerator import AcceleratorController, AcceleratorState
from .config import BatchSimulationConfig, NetworkConfig, SimulationConfig
from .network import NetworkFactory, smooth_size_sequence
from .payoff import PublicGoodsPayoffCalculator
from .strategy import StrategyInitializer, StrategyUpdater, average_profit, cooperation_ratio
from .simulation import PublicGoodsGameSimulator, RoundRecord, SimulationResult

__all__ = [
    "AcceleratorController",
    "AcceleratorState",
    "BatchSimulationConfig",
    "NetworkConfig",
    "SimulationConfig",
    "NetworkFactory",
    "smooth_size_sequence",
    "PublicGoodsPayoffCalculator",
    "StrategyInitializer",
    "StrategyUpdater",
    "average_profit",
    "cooperation_ratio",
    "PublicGoodsGameSimulator",
    "RoundRecord",
    "SimulationResult",
]
