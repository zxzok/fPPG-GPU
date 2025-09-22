
"""公共物品博弈（PGG）仿真主程序。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from pgg_gaming import (
    AcceleratorController,
    NetworkConfig,
    PublicGoodsGameSimulator,
    SimulationConfig,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行单网络公共物品博弈仿真，支持 GPU 加速与策略偏好设置。"
    )
    parser.add_argument("--topology", choices=["ba", "ws", "er", "regular", "tree"], default="ba")
    parser.add_argument("--size", type=int, default=100, help="网络节点数量")
    parser.add_argument("--degree", type=int, default=4, help="期望平均度")
    parser.add_argument("--rounds", type=int, default=200, help="博弈迭代轮数")
    parser.add_argument("--synergy", type=float, default=3.0, help="协同系数 r")
    parser.add_argument(
        "--initial-cooperation",
        type=float,
        default=0.5,
        help="初始化时合作者比例的期望值",
    )
    parser.add_argument("--policy-alpha-g", type=float, default=0.0, help="政策导向影响强度")
    parser.add_argument("--policy-alpha-i", type=float, default=1.0, help="个体收益权重")
    parser.add_argument("--enable-policy", action="store_true", help="开启政策导向学习模式")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子")
    parser.add_argument("--gpu-device", type=str, default=None, help="指定 CUDA 设备，如 cuda:0")
    parser.add_argument("--force-cpu", action="store_true", help="强制关闭 GPU 加速")
    parser.add_argument("--output", type=Path, help="结果导出为 Excel 文件的路径")
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_simulation_config(args: argparse.Namespace) -> SimulationConfig:
    network_cfg = NetworkConfig(
        topology=args.topology,
        size=args.size,
        mean_degree=args.degree,
    )
    return SimulationConfig(
        network=network_cfg,
        rounds=args.rounds,
        synergy_factor=args.synergy,
        initial_cooperation=args.initial_cooperation,
        policy_alpha_g=args.policy_alpha_g,
        policy_alpha_i=args.policy_alpha_i,
        enable_policy_bias=args.enable_policy,
        random_seed=args.seed,
    )


def prepare_accelerator(args: argparse.Namespace) -> AcceleratorController:
    controller = AcceleratorController()
    if args.force_cpu:
        controller.disable()
    elif args.gpu_device:
        controller.enable(args.gpu_device)
    return controller


def run_simulation(args: argparse.Namespace) -> None:
    config = create_simulation_config(args)
    controller = prepare_accelerator(args)

    simulator = PublicGoodsGameSimulator(config, accelerator=controller)
    result = simulator.run()

    logging.info("GPU 使用状态：%s", controller.device)
    logging.info("最终合作者比例：%.4f", result.records[-1].cooperation_ratio)
    logging.info("最终平均收益：%.4f", result.records[-1].average_profit)

    if args.output:
        dataframe = result.to_dataframe()
        dataframe.to_excel(args.output, index=False)
        logging.info("结果已导出到 %s", args.output)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging()
    run_simulation(args)


if __name__ == "__main__":
    main()

