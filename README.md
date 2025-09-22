# fPPG-GPU 项目说明

本项目实现了公共物品博弈（Public Goods Game, PGG）的 GPU 加速仿真。
为了便于维护与扩展，本次重构完成了如下工作：

- 建立 `pgg_gaming` 模块化架构，划分 GPU 控制、网络生成、策略更新、收益计算与仿真调度等子模块；
- 通过类型注解与中文文档补充主要接口说明；
- 新增命令行入口 `GPU程序调试/PGG_A.py`，可以通过参数灵活配置网络拓扑、协同系数、轮数、政策导向等；
- 保留 `gaming_models.py` 旧接口，内部转调新模块，避免历史脚本失效。

## 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt  # 如有需要
   ```

2. 运行仿真
   ```bash
   python GPU程序调试/PGG_A.py --topology ba --size 200 --rounds 500 \
       --synergy 3.5 --initial-cooperation 0.6 --enable-policy \
       --policy-alpha-g 0.2 --policy-alpha-i 0.8 --output result.xlsx
   ```

3. GPU 控制
   - `--gpu-device cuda:0` 指定使用的 CUDA 设备；
   - `--force-cpu` 强制回退到 CPU 模式；
   - 也可使用环境变量 `PGG_GPU_DEVICE` 与 `PGG_FORCE_CPU` 全局控制。

## 目录结构

```
GPU程序调试/
├── PGG_A.py                # 命令行入口
├── gaming_models.py        # 兼容旧接口
├── pgg_gaming/             # 新的核心模块
│   ├── accelerator.py      # GPU 控制器
│   ├── config.py           # 参数配置数据类
│   ├── network.py          # 网络生成与规模平滑
│   ├── payoff.py           # 收益计算（CPU/GPU）
│   ├── simulation.py       # 仿真调度器
│   └── strategy.py         # 策略初始化与学习规则
└── 程序及GPU情况说明.txt    # 架构与 GPU 使用说明
```

欢迎在 `issues` 中反馈问题或提出改进建议。
