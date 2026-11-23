"""geostat — 带趋势项的空间插值（泛克里金）核心库。

模块划分：
- covariance   相关模型（指数/高斯/Matérn-3/2/球状）与协方差装配
- geom         屏障多边形、可见性图测地距离、网格图距离
- preprocessing 重复坐标 / 近重合点 / 数值病态的分别处理与报告
- kriging      泛克里金系统（趋势 + 协方差），预测、方差、权重
- estimation   协方差参数估计（剖面 REML），仅用训练数据
- blocks       连续空间块划分（批次不拆分）与块交叉验证
- design       考虑相关性与费用的加点建议
- simulate     已知随机场模拟（测试与合成测量）
- dataset      数据集加载与校验
- model        顶层模型对象：拟合/预测/版本管理
"""

__all__ = [
    "covariance",
    "geom",
    "preprocessing",
    "kriging",
    "estimation",
    "blocks",
    "design",
    "simulate",
    "dataset",
    "model",
]
