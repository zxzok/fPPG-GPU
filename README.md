# fPPG-GPU 使用手册

## 1. 项目概述
fPPG-GPU 是一个用于模拟公共物品博弈（Public Goods Game, PGG）的研究工具，重点支持 GPU 加速的收益计算以及灵活的策略演化配置。项目以 `pgg_gaming` 模块为核心，围绕网络生成、策略初始化与学习、收益结算和仿真调度等环节提供了高度模块化的实现，同时保留了旧版脚本的接口兼容层，方便历史实验延续。

## 2. 核心特性
- **GPU 加速的收益计算**：通过 `AcceleratorController` 自动侦测 CUDA 环境并选择 GPU/CPU 计算路径，必要时可通过命令行或环境变量强制切换。【`GPU程序调试/pgg_gaming/accelerator.py`】
- **策略演化全流程建模**：`PublicGoodsGameSimulator` 集成了网络构建、策略初始化、收益计算与策略更新，实现从单轮迭代到完整仿真的自动化流程。【`GPU程序调试/pgg_gaming/simulation.py`】
- **政策导向学习机制**：在经典费米学习规则基础上，可叠加外部政策偏好（参数 `alpha_g`、`alpha_i`），用于探索协作行为的导向控制效果。【`GPU程序调试/pgg_gaming/strategy.py`】
- **命令行一键运行**：`GPU程序调试/PGG_A.py` 提供丰富的命令行参数，可配置网络类型、规模、协同系数、轮数、随机种子及政策偏好，并支持结果导出到 Excel。【`GPU程序调试/PGG_A.py`】
- **兼容旧版接口**：`gaming_models.py` 将旧函数调用映射至新模块，历史脚本无需修改即可复用 GPU 能力与策略算法。【`GPU程序调试/gaming_models.py`】
- **网络生成与分析工具**：随附 `net_creat.py`、`net_analysis.py`、`net_save_load.py` 等脚本，可批量生成 BA/WS/ER/规则/树形等网络、导入真实社会网络，并输出统计指标到 Excel。【`GPU程序调试` 目录下相关脚本】

## 3. 环境要求
- 建议使用 **Python 3.10 及以上版本**（与 `networkx>=3.0` 兼容）。
- 基础依赖：`networkx`、`numpy`、`pandas`。
- 可选依赖：`torch`（仅在需要 GPU 加速时安装）。

安装方式：
```bash
pip install -r requirements.txt
# 需要 GPU 时另外安装 PyTorch： https://pytorch.org/get-started/locally/
```

## 4. 目录结构
```
GPU程序调试/
├── PGG_A.py                 # 命令行入口，负责解析参数并运行仿真
├── gaming_models.py         # 兼容旧接口的适配层
├── pgg_gaming/              # 新版核心模块
│   ├── accelerator.py       # GPU 设备检测与启停控制
│   ├── config.py            # 网络与仿真参数数据类
│   ├── network.py           # 多种网络拓扑生成与规模平滑
│   ├── payoff.py            # 收益计算（CPU/GPU 双实现）
│   ├── simulation.py        # 仿真调度与结果记录
│   └── strategy.py          # 策略初始化、费米学习、政策偏好
├── net_creat.py             # 批量生成/拼接异质网络、导出统计
├── net_analysis.py          # 网络拓扑特征分析工具
├── net_save_load.py         # 网络保存/加载，含真实网络读取
├── net_visualization.py     # 网络可视化示例
├── write2excel.py           # Excel 写入工具函数
├── realnets/                # 示例真实网络数据（Female、Male、Nyangatom）
└── 程序及GPU情况说明.txt     # 架构与 GPU 功能补充说明
```

## 5. 快速开始
1. **准备环境**：安装依赖并（如需）配置 CUDA 驱动。
2. **运行仿真**：
   ```bash
   python GPU程序调试/PGG_A.py \
       --topology ba \
       --size 200 \
       --degree 4 \
       --rounds 500 \
       --synergy 3.5 \
       --initial-cooperation 0.6 \
       --enable-policy \
       --policy-alpha-g 0.2 \
       --policy-alpha-i 0.8 \
       --gpu-device cuda:0 \
       --output results.xlsx
   ```
3. **查看日志**：终端将输出所使用的设备、最终协作比例与平均收益。
4. **分析结果**：若提供 `--output` 参数，会生成包含每轮 `cooperation_ratio`、`average_profit` 的 Excel 文件。

## 6. 命令行参数一览
| 参数 | 说明 |
| --- | --- |
| `--topology` | 网络类型，可选 `ba`、`ws`、`er`、`regular`、`tree`。|
| `--size` / `--degree` | 网络节点数与期望平均度。|
| `--rounds` | 仿真轮数。|
| `--synergy` | 协同系数 *r*。|
| `--initial-cooperation` | 初始化时合作者比例期望值。|
| `--policy-alpha-g` / `--policy-alpha-i` | 政策导向强度与个体收益权重。|
| `--enable-policy` | 是否启用政策偏好学习。|
| `--seed` | 随机数种子（影响网络生成、策略初始化与学习顺序）。|
| `--gpu-device` | 指定 CUDA 设备（如 `cuda:0`）。|
| `--force-cpu` | 强制关闭 GPU 加速。|
| `--output` | 导出结果到 Excel 文件。|

## 7. 仿真流程解析
1. **网络生成**：`NetworkFactory` 根据拓扑类型与参数构建网络，并设置初始节点属性。【`pgg_gaming/network.py`】
2. **策略初始化**：`StrategyInitializer` 按设定概率为每个节点赋予合作/背叛策略并随机生成偏好值。【`pgg_gaming/strategy.py`】
3. **收益计算**：`PublicGoodsPayoffCalculator` 以协同系数 *r* 计算每个团体的收益，对应节点累积收益，可自动在 CPU/GPU 间切换。【`pgg_gaming/payoff.py`】
4. **策略更新**：`StrategyUpdater` 依据费米规则或政策扩展版本调整策略，记录边上学习概率。 【`pgg_gaming/strategy.py`】
5. **数据记录**：`SimulationResult` 储存每轮协作比例与平均收益，并支持转换为 `pandas.DataFrame` 导出。【`pgg_gaming/simulation.py`】

## 8. 在 Python 中调用模块
除命令行外，也可直接在 Python 代码中调用：
```python
from pathlib import Path
from pgg_gaming import AcceleratorController, NetworkConfig, SimulationConfig, PublicGoodsGameSimulator

network_cfg = NetworkConfig(topology="ws", size=150, mean_degree=6)
sim_cfg = SimulationConfig(
    network=network_cfg,
    rounds=300,
    synergy_factor=3.2,
    initial_cooperation=0.55,
    enable_policy_bias=True,
    policy_alpha_g=0.1,
    policy_alpha_i=0.9,
    random_seed=42,
)

accelerator = AcceleratorController()
accelerator.enable("cuda:0")  # 若无 GPU 可省略或改用 accelerator.disable()

simulator = PublicGoodsGameSimulator(sim_cfg, accelerator=accelerator)
result = simulator.run()

print("最终协作率", result.records[-1].cooperation_ratio)
result.to_dataframe().to_excel(Path("custom_run.xlsx"), index=False)
```

## 9. GPU 配置与环境变量
- `--gpu-device cuda:0`：在命令行中指定 CUDA 设备。
- `--force-cpu`：临时关闭 GPU，加速器回退至 CPU 逻辑。
- 环境变量：
  - `PGG_GPU_DEVICE`：全局默认 GPU 设备（值为 `auto` 时自动检测可用设备）。
  - `PGG_FORCE_CPU`：设置为 `1/true/on` 等真值时，禁用 GPU。

若 PyTorch 未安装或 CUDA 不可用，程序会自动降级为 CPU 版本，并在日志中提示。

## 10. 兼容旧版脚本
- 旧函数如 `calc_profit_PGG`、`game_stra_learn`、`learning_Pr_sigmod` 等仍可在 `gaming_models.py` 中调用，内部会转向新版模块实现。
- 可通过 `set_gpu_mode(True, "cuda:0")` 控制 GPU 状态，或使用 `calc_learn_Probabilty` 等函数直接在旧脚本中计算学习概率。

## 11. 网络生成与真实网络支持
- `net_creat.py`：批量生成同质或异质网络，支持 BA/WS/ER/REG/TREE 等拓扑，可结合 `write2excel.py` 写入统计结果；亦可创建小规模网络以校验收益算法。
- `net_analysis.py`：计算网络的平均度、直径、平均最短路径、聚类系数及度分布等指标，并输出至 Excel。
- `net_save_load.py`：支持保存/读取网络结构，可直接加载 `realnets/` 中的 Female、Male、Nyangatom 社会网络。

## 12. 常见问题
1. **未安装 PyTorch 是否可以运行？** 可以，收益计算会自动使用 CPU 实现，但在大规模网络下运行速度会受限。
2. **如何加速批量实验？** 可组合 `BatchSimulationConfig`（位于 `config.py`）与自定义脚本遍历网络规模，或借助 `net_creat.py` 预生成网络后循环调用仿真器。
3. **Excel 导出失败？** 确认已安装 `pandas` 与对应的 Excel 写入依赖（如 `openpyxl`）。若文件被占用，请关闭对应的 Excel 文件后重试。

---
欢迎在 Issues 中反馈问题或贡献改进建议。祝实验顺利！
