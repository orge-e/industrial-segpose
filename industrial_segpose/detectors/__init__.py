"""专用检测器子包：保存面向特定工况、实现统一检测接口的算法。"""

from .fluorescent import FluorescentTextileDetector, FluorescentTextileParameters

__all__ = ["FluorescentTextileDetector", "FluorescentTextileParameters"]
