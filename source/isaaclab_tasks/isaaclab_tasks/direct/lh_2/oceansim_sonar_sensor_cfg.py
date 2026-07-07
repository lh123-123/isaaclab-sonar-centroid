from dataclasses import dataclass
from typing import Literal

from isaaclab.utils import configclass


@configclass
class OceanSimSonarSensorCfg:
    """OceanSim高保真声呐传感器配置
    
    基于Oculus M370s/MT370s/MD370s真实参数配置
    """

    min_range: float = 0.5
    """最小探测距离（米）"""

    max_range: float = 8.0
    """最大探测距离（米）"""

    range_res: float = 0.05
    """距离分辨率（米）"""

    hori_fov: float = 90.0
    """水平视场角（度）"""

    vert_fov: float = 45.0
    """垂直视场角（度）"""

    angular_res: float = 2.0
    """角度分辨率（度）"""

    hori_res: int = 128
    """水平分辨率（像素）"""

    height: int = 64
    """声呐图像高度（像素）"""

    width: int = 128
    """声呐图像宽度（像素）"""

    beam_count: int = 128
    """波束数量"""

    beam_topk: int = 5
    """每个beam列用于距离估计的top-k采样数（加权中心法）"""

    beam_topk_min_weight_sum: float = 1.0e-4
    """top-k总权重下限；低于阈值时回退到argmax距离估计"""

    include_unlabelled: bool = True
    """是否包含未标记物体"""

    binning_method: str = "sum"
    """强度累积方法：'sum' 或 'mean'"""

    normalizing_method: str = "range"
    """归一化方法：'all'（全局最大值）或 'range'（每距离最大值）"""

    attenuation: float = 0.1
    """距离衰减系数"""

    gau_noise_param: float = 0.2
    """高斯噪声乘数"""

    ray_noise_param: float = 0.05
    """瑞利噪声尺度因子"""

    intensity_offset: float = 0.0
    """归一化后的强度偏移"""

    intensity_gain: float = 1.0
    """归一化后的强度增益"""

    sonar_initialize_viewport: bool = False
    """传给 OceanSim ``sonar_initialize(viewport=...)``。
    False：走离屏渲染管线（训练/headless 更稳，减轻 ``Replicator_pointcloud`` KeyError）。
    True：绑定视口 SDG（仅当你必须在 GUI 里依赖视口后处理时再开）。
    """

    use_replicator_orchestrator_recovery: bool = False
    """声呐 ``scan`` / ``make_sonar_data`` 遇 KeyError 时是否调用 ``rep.orchestrator.step()`` 再试。
    在部分机器上会长时间阻塞主线程（界面像卡死），默认关闭；仅保留 scan 内建重试。"""