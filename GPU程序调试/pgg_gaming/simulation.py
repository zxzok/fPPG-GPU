"""公共物品博弈仿真调度模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import random
from typing import List

import networkx as nx
import numpy as np

from .accelerator import AcceleratorController
from .config import SimulationConfig
from .network import NetworkFactory
from .payoff import PublicGoodsPayoffCalculator
from .strategy import (
    StrategyInitializer,
    StrategyUpdater,
    average_profit,
    cooperation_ratio,
)


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RoundRecord:
    """单轮博弈的关键统计信息。"""

    round_index: int
    cooperation_ratio: float
    average_profit: float


@dataclass(slots=True)
class SimulationResult:
    """封装仿真过程产生的所有数据。"""

    records: List[RoundRecord] = field(default_factory=list)
    device: str = "cpu"

    def to_dataframe(self):
        import pandas as pd  # 延迟导入，避免在无需导出时强制依赖

        return pd.DataFrame(
            {
                "round": [record.round_index for record in self.records],
                "cooperation_ratio": [record.cooperation_ratio for record in self.records],
                "average_profit": [record.average_profit for record in self.records],
            }
        )


class PublicGoodsGameSimulator:
    """公共物品博弈主仿真器。"""

    def __init__(self, config: SimulationConfig, accelerator: AcceleratorController | None = None) -> None:
        self.config = config
        self.accelerator = accelerator or AcceleratorController()
        self.payoff_calculator = PublicGoodsPayoffCalculator(
            investment=config.cooperative_investment,
            accelerator=self.accelerator,
        )
        self.initializer = StrategyInitializer(
            cooperative_probability=config.initial_cooperation,
            seed=config.random_seed,
        )
        self.updater = StrategyUpdater(
            temperature=config.learning_temperature,
            seed=config.random_seed,
        )
        self.network_factory = NetworkFactory(seed=config.random_seed)

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        self.graph: nx.Graph | None = None

    # ------------------------------------------------------------------
    def setup(self) -> None:
        self.graph = self.network_factory.create(self.config.network)
        self.initializer.apply(self.graph)

    def run(self) -> SimulationResult:
        if self.graph is None:
            self.setup()
        assert self.graph is not None

        records: List[RoundRecord] = []
        use_policy = self.config.enable_policy_bias and (
            self.config.policy_alpha_g != 0.0 or self.config.policy_alpha_i != 1.0
        )

        for round_index in range(1, self.config.rounds + 1):
            self.payoff_calculator.calculate(self.graph, self.config.synergy_factor)
            if use_policy:
                self.updater.update_with_policy(
                    self.graph,
                    alpha_g=self.config.policy_alpha_g,
                    alpha_i=self.config.policy_alpha_i,
                )
            else:
                self.updater.update(self.graph)

            record = RoundRecord(
                round_index=round_index,
                cooperation_ratio=cooperation_ratio(self.graph),
                average_profit=average_profit(self.graph),
            )
            records.append(record)

        return SimulationResult(records=records, device=self.accelerator.device)


__all__ = [
    "PublicGoodsGameSimulator",
    "RoundRecord",
    "SimulationResult",
]
