"""公共物品博弈（PGG）模型旧接口的兼容层。"""

from __future__ import annotations

import logging
import math
import random
from typing import Iterable, List

import networkx as nx

from pgg_gaming import (
    AcceleratorController,
    PublicGoodsPayoffCalculator,
    StrategyInitializer,
    StrategyUpdater,
    average_profit as _average_profit,
    smooth_size_sequence,
)

_LOGGER = logging.getLogger(__name__)

_CONTROLLER = AcceleratorController()
_PAYOFF = PublicGoodsPayoffCalculator(accelerator=_CONTROLLER)
_INITIALIZER = StrategyInitializer()
_UPDATER = StrategyUpdater()
_DEFAULT_TEMPERATURE = 0.5


def set_gpu_mode(enable: bool, device: str | None = None) -> bool:
    if enable:
        return _CONTROLLER.enable(device)
    _CONTROLLER.disable()
    return False


def is_gpu_enabled() -> bool:
    return _CONTROLLER.is_enabled


def get_accelerator_device_name() -> str:
    return _CONTROLLER.device


def calc_profit_PGG(net: nx.Graph, r: float) -> None:
    _PAYOFF.calculate(net, r)


def game_stra_learn(net: nx.Graph, r: float = 2.0) -> None:  # noqa: ARG001 兼容旧接口
    _UPDATER.update(net)


def game_stra_learn_withPrefer(
    net: nx.Graph,
    r: float = 2.0,  # noqa: ARG001 保留旧参数
    alpha_G: float = 0.0,
    alpha_I: float = 1.0,
) -> None:
    _UPDATER.update_with_policy(net, alpha_g=alpha_G, alpha_i=alpha_I)


def learning_Pr_sigmod(fnode: float, fnei: float, temperature: float = _DEFAULT_TEMPERATURE) -> float:
    delta = fnode - fnei
    if delta >= 100:
        return 0.0
    if delta <= -100:
        return 1.0
    temperature = max(temperature, 1e-6)
    return 1.0 / (1.0 + math.exp(delta / temperature))


def calc_learn_Probabilty(net: nx.Graph, r: float, alpha_G: float, alpha_I: float) -> None:  # noqa: ARG001
    snapshot = {node: net.nodes[node].get("strategy", net.nodes[node].get("select", 0)) for node in net.nodes}
    profits = {node: net.nodes[node].get("profit", 0.0) for node in net.nodes}

    for u, v in net.edges:
        base_uv = learning_Pr_sigmod(profits[u], profits[v])
        base_vu = learning_Pr_sigmod(profits[v], profits[u])

        pref_u = net.nodes[u].get("preference", 0.0)
        pref_v = net.nodes[v].get("preference", 0.0)

        prob_uv = alpha_I * base_uv + alpha_G * pref_u * (1 if snapshot[v] == 1 else -1)
        prob_vu = alpha_I * base_vu + alpha_G * pref_v * (1 if snapshot[u] == 1 else -1)

        prob_uv = min(max(prob_uv, 0.0), 1.0)
        prob_vu = min(max(prob_vu, 0.0), 1.0)

        net[u][v]["study_probability"] = {u: prob_uv, v: prob_vu}


def calc_C_num(net: nx.Graph) -> int:
    return sum(int(net.nodes[node].get("strategy", net.nodes[node].get("select", 0))) for node in net.nodes)


def calc_ave_profit(net: nx.Graph) -> float:
    return _average_profit(net)


def set_gaming_role(net: nx.Graph, cooperate_P: float, betray_P: float) -> List[int]:
    if cooperate_P < 0 or betray_P < 0:
        raise ValueError("概率不能小于 0")

    total = net.number_of_nodes()
    count_C = 0
    count_B = 0

    for node in net.nodes:
        rnd = random.random()
        if rnd < cooperate_P:
            role = "C"
            count_C += 1
        elif rnd < cooperate_P + betray_P:
            role = "B"
            count_B += 1
        else:
            role = "N"
        net.nodes[node]["role"] = role

    return [count_C, count_B, total - count_C - count_B]


def init_game_strategy(net: nx.Graph) -> int:
    _INITIALIZER.apply(net)
    return calc_C_num(net)


def init_game_strategy_withRole(net: nx.Graph) -> int:
    cooperators = 0
    for node in net.nodes:
        role = net.nodes[node].get("role", "N")
        if role == "C":
            strategy = 1
        elif role == "B":
            strategy = 0
        else:
            strategy = random.randint(0, 1)
        net.nodes[node]["strategy"] = strategy
        net.nodes[node]["select"] = strategy
        net.nodes[node]["profit"] = 0.0
        net.nodes[node].setdefault("preference", random.random())
        cooperators += strategy
    return cooperators


def smoothing_net_size_changes(sizes_arr: Iterable[int], maxScale: float = 0.5) -> List[int]:
    return smooth_size_sequence(sizes_arr, max_scale=maxScale)


__all__ = [name for name in globals() if not name.startswith("_")]
