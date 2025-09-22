"""GPU 加速控制模块。

该模块负责统一管理 PyTorch CUDA 环境的检测、启用与关闭逻辑，
供公共物品博弈的收益计算调用。设计目标如下：

1. 统一封装环境变量、日志提示与错误处理，避免业务代码中充斥
   重复的 CUDA 判断语句；
2. 在无法使用 GPU 的情况下自动退回 CPU，实现“尽力而为”的体验；
3. 提供清晰的中文文档与类型注解，方便后续维护人员快速上手。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Optional

try:  # PyTorch 是可选依赖，故采用 try/except 形式导入
    import torch
except ImportError:  # pragma: no cover - 运行环境缺乏 PyTorch 时进入该分支
    torch = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AcceleratorState:
    """记录 GPU 加速器当前的状态信息。"""

    enabled: bool = False
    device: Optional[str] = None


class AcceleratorController:
    """GPU 加速控制器。"""

    def __init__(self) -> None:
        self._state = AcceleratorState()
        self._torch = torch
        self._force_cpu = os.environ.get("PGG_FORCE_CPU", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if self._force_cpu:
            _LOGGER.info("检测到环境变量 PGG_FORCE_CPU，强制关闭 GPU 加速。")
            return

        if self._torch is None:
            _LOGGER.info("未检测到 PyTorch，收益计算将使用 CPU 版本。")
            return

        requested = os.environ.get("PGG_GPU_DEVICE", "auto")
        self.enable(None if requested == "auto" else requested)

    @property
    def is_enabled(self) -> bool:
        """返回当前是否使用 GPU 进行收益计算。"""

        return self._state.enabled

    @property
    def device(self) -> str:
        """返回 GPU 设备名称；如未启用则返回 ``"cpu"``。"""

        return self._state.device or "cpu"

    def enable(self, device: Optional[str] = None) -> bool:
        """启用 GPU 加速。"""

        if self._force_cpu:
            _LOGGER.info("当前处于强制 CPU 模式，忽略启用 GPU 的请求。")
            return False

        if self._torch is None:
            _LOGGER.warning("PyTorch 未安装，无法启用 GPU。")
            self._state = AcceleratorState(enabled=False, device=None)
            return False

        chosen_device = device or os.environ.get("PGG_GPU_DEVICE", "cuda")

        if not self._torch.cuda.is_available():
            _LOGGER.warning("CUDA 设备不可用，收益计算将继续使用 CPU。")
            self._state = AcceleratorState(enabled=False, device=None)
            return False

        try:
            torch_device = self._torch.device(chosen_device)
            if torch_device.type != "cuda":
                raise RuntimeError(f"仅支持 CUDA 设备，收到 {chosen_device!r}")

            self._torch.zeros(1, device=torch_device)
        except Exception as exc:  # pragma: no cover - 依赖外部 CUDA 环境
            _LOGGER.warning(
                "GPU 初始化失败（设备 %s）：%s，自动回退至 CPU。",
                chosen_device,
                exc,
            )
            self._state = AcceleratorState(enabled=False, device=None)
            return False

        self._state = AcceleratorState(enabled=True, device=str(torch_device))
        _LOGGER.info("成功启用 GPU 设备 %s。", self._state.device)
        return True

    def disable(self) -> None:
        """关闭 GPU，加速器回退到 CPU 模式。"""

        if self._state.enabled:
            _LOGGER.info("关闭 GPU 加速，后续收益计算将使用 CPU。")
        self._state = AcceleratorState(enabled=False, device=None)


__all__ = ["AcceleratorController", "AcceleratorState"]
