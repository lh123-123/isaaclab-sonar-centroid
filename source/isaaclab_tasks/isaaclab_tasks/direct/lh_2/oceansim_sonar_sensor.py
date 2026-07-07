from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .oceansim_sonar_sensor_cfg import OceanSimSonarSensorCfg


class OceanSimSonarSensor:
    """OceanSim高保真声呐传感器包装类
    
    基于OceanSim的ImagingSonarSensor，适配IsaacLab环境使用。
    
    特性：
    - 基于Oculus M370s/MT370s/MD370s真实参数
    - GPU加速声呐图像生成（Warp kernels）
    - 支持声呐图像和波束数据
    - 符合PROJECT_GUIDE.md要求的高保真仿真
    
    注意：
    - 必须在启动Isaac Sim时加载OceanSim扩展
    - 必须在场景加载完成后调用initialize()
    - 必须调用close()来释放GPU资源
    """

    def __init__(self, cfg: OceanSimSonarSensorCfg, num_envs: int = 1, device: str = "cuda:0"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        self._sensors = []
        self._initialized = False
        self._asset_path_registered = False
        
        # 初始化时使用配置的尺寸，实际尺寸会在第一次更新时确定
        self._sonar_image = torch.zeros(
            num_envs, 
            cfg.height, 
            cfg.width, 
            device=device
        )
        self._beam_data = torch.zeros(
            num_envs, 
            cfg.beam_count, 
            3,  # [angle, range, intensity]
            device=device
        )
        
        # 用于错误恢复的状态跟踪
        self._last_valid_sonar_image = None
        self._last_valid_beam_data = None
        
        # 原始声呐图尺寸（OceanSim 输出可能与 cfg 不一致）
        self._actual_height = None
        self._actual_width = None
        self._resize_notice_logged = False
        # Cache beam geometry tensors used every step to avoid repeated allocations.
        self._beam_cache_width = None
        self._beam_cache_fov = None
        self._beam_cache_angles = None
        self._beam_cache_center_x = None

    def initialize(self, env_idx: int, prim_path: str):
        """初始化单个环境的声呐传感器
        
        Args:
            env_idx: 环境索引
            prim_path: 机器人USD路径
        """
        while env_idx >= len(self._sensors):
            self._sensors.append(None)
        
        sensor = None
        try:
            import sys
            import os
            import importlib
            import importlib.util
            import types
            
            from isaaclab_tasks.direct.nonar2.oceansim_ext_utils import (
                ensure_oceansim_kit_extensions,
                oceansim_extension_root,
                resolve_isaacsim_exts_user_folder,
            )

            ext_folder = resolve_isaacsim_exts_user_folder()
            
            # 注册OceanSim资源路径（仅一次）
            if not self._asset_path_registered:
                import json
                oceansim_root = os.path.join(ext_folder, "isaacsim.oceansim")
                asset_path = os.path.join(oceansim_root, "data")
                json_path = os.path.join(oceansim_root, "isaacsim", "oceansim", "utils", "asset_path.json")
                
                if os.path.exists(asset_path) and os.path.exists(json_path):
                    json_data = {"asset_path": os.path.abspath(asset_path)}
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    self._asset_path_registered = True
                    print(f"OceanSim asset path registered: {json_data['asset_path']}")
            
            # 构建完整的isaacsim.oceansim模块结构
            oceansim_ext_path = oceansim_extension_root(ext_folder)
            oceansim_isaacsim_path = os.path.join(oceansim_ext_path, "isaacsim")
            oceansim_oceansim_path = os.path.join(oceansim_isaacsim_path, "oceansim")
            oceansim_sensors_path = os.path.join(oceansim_oceansim_path, "sensors")
            oceansim_utils_path = os.path.join(oceansim_oceansim_path, "utils")
            
            # 添加路径到sys.path
            for path in [oceansim_isaacsim_path, oceansim_oceansim_path]:
                if path not in sys.path:
                    sys.path.insert(0, path)

            try:
                ensure_oceansim_kit_extensions(ext_folder)
            except Exception as e:
                raise ImportError(
                    "Failed to enable required Isaac Sim extensions "
                    "('isaacsim.sensors.camera', 'isaacsim.oceansim') for OceanSim."
                ) from e
            
            # 重要：不能伪造/覆盖顶级 isaacsim 模块，否则会破坏
            # ImagingSonarSensor.py 内部 `from isaacsim.sensors.camera import Camera`。
            try:
                isaacsim_module = importlib.import_module("isaacsim")
            except Exception as e:
                raise ImportError(
                    "Failed to import top-level 'isaacsim' package. "
                    "Please ensure Isaac Sim Python environment is active."
                ) from e

            if "isaacsim.oceansim" not in sys.modules:
                oceansim_module = types.ModuleType("oceansim")
                sys.modules["isaacsim.oceansim"] = oceansim_module
                setattr(isaacsim_module, "oceansim", oceansim_module)
            else:
                oceansim_module = sys.modules["isaacsim.oceansim"]
            
            # 导入utils模块
            utils_module_path = os.path.join(oceansim_utils_path, "ImagingSonar_kernels.py")
            if os.path.exists(utils_module_path):
                utils_spec = importlib.util.spec_from_file_location("isaacsim.oceansim.utils.ImagingSonar_kernels", utils_module_path)
                if utils_spec is None:
                    raise ImportError(f"Failed to load module spec for {utils_module_path}")
                utils_module = importlib.util.module_from_spec(utils_spec)
                sys.modules['isaacsim.oceansim.utils.ImagingSonar_kernels'] = utils_module
                utils_spec.loader.exec_module(utils_module)
                setattr(oceansim_module, 'utils', types.ModuleType('utils'))
                setattr(oceansim_module.utils, 'ImagingSonar_kernels', utils_module)
            
            # 导入sensors模块
            sensor_module_path = os.path.join(oceansim_sensors_path, "ImagingSonarSensor.py")
            if os.path.exists(sensor_module_path):
                sensor_spec = importlib.util.spec_from_file_location("isaacsim.oceansim.sensors.ImagingSonarSensor", sensor_module_path)
                if sensor_spec is None:
                    raise ImportError(f"Failed to load module spec for {sensor_module_path}")
                sensor_module = importlib.util.module_from_spec(sensor_spec)
                sys.modules['isaacsim.oceansim.sensors.ImagingSonarSensor'] = sensor_module
                sensor_spec.loader.exec_module(sensor_module)
                setattr(oceansim_module, 'sensors', types.ModuleType('sensors'))
                setattr(oceansim_module.sensors, 'ImagingSonarSensor', sensor_module.ImagingSonarSensor)
                ImagingSonarSensor = sensor_module.ImagingSonarSensor
            else:
                raise FileNotFoundError(f"ImagingSonarSensor.py not found at {sensor_module_path}")
            
            sensor_prim_path = f"{prim_path}/SonarSensor"
            sensor = ImagingSonarSensor(  # type: ignore
                prim_path=sensor_prim_path,
                name=f"sonar_{env_idx}",
                min_range=self.cfg.min_range,
                max_range=self.cfg.max_range,
                range_res=self.cfg.range_res,
                hori_fov=self.cfg.hori_fov,
                vert_fov=self.cfg.vert_fov,
                angular_res=self.cfg.angular_res,
                hori_res=self.cfg.hori_res,
                render_product_path=None,  # 让Camera类自动创建
                physics_sim_view=None  # 在这个阶段physics_sim_view还不可用
            )
        except ImportError as e:
            raise ImportError(
                "OceanSim extension not found. Please install OceanSim extension from: "
                "https://github.com/NVIDIA-OceanSim/OceanSim\n\n"
                "Make sure to load the extension when starting Isaac Sim. "
                "You can do this by:\n"
                "1. Running: ./scripts/train_lh_2.sh (uses correct extension path)\n"
                "2. Or setting the environment variable: ISAACSIM_EXTENSIONS_FOLDER=/path/to/extsUser\n"
                f"\nOriginal error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"[OceanSim][env {env_idx}] Failed to create sonar sensor at prim '{prim_path}/SonarSensor': {e}"
            ) from e
        
        if sensor is None:
            raise RuntimeError(f"[OceanSim][env {env_idx}] Sonar sensor object is None after creation.")
        
        try:
            # 详细日志：检查传感器创建状态
            print(f"[Sensor {env_idx}] Creating ImagingSonarSensor...")
            print(f"[Sensor {env_idx}] Prim path: {sensor_prim_path}")
            print(f"[Sensor {env_idx}] Config: hori_res={self.cfg.hori_res}, vert_res={self.cfg.hori_res // (self.cfg.hori_fov/self.cfg.vert_fov)}")
            
            # 2. 检查渲染产品路径（在sonar_initialize前验证）
            print(f"[Sensor {env_idx}] Verifying render product path...")
            if not hasattr(sensor, '_render_product_path'):
                raise RuntimeError(
                    f"[OceanSim][env {env_idx}] Missing _render_product_path on ImagingSonarSensor."
                )
                
            if sensor._render_product_path is None:
                raise RuntimeError(
                    f"[OceanSim][env {env_idx}] _render_product_path is None. Camera init likely failed."
                )
                
            print(f"[Sensor {env_idx}] Render product path: {sensor._render_product_path}")
            
            # 3. 避免重复初始化
            if hasattr(sensor, 'pointcloud_annot'):
                print(f"[Sensor {env_idx}] WARNING: Sensor already initialized, skipping sonar_initialize()")
                self._sensors[env_idx] = sensor
                return
            
            # 1. 确保正确的初始化顺序：在父类__init__完成后调用sonar_initialize()
            print(f"[Sensor {env_idx}] Initializing sonar sensor (first time)...")
            
            if self.cfg.include_unlabelled:
                sensor.sonar_initialize(
                    include_unlabelled=True,
                    if_array_copy=True,
                    viewport=self.cfg.sonar_initialize_viewport,
                    output_dir=None,
                )
            else:
                sensor.sonar_initialize(
                    include_unlabelled=False,
                    if_array_copy=True,
                    viewport=self.cfg.sonar_initialize_viewport,
                    output_dir=None,
                )
            
            # 验证注释器创建成功
            if not hasattr(sensor, 'pointcloud_annot'):
                raise RuntimeError(
                    f"[OceanSim][env {env_idx}] pointcloud_annot not created after sonar_initialize()."
                )
            
            if not hasattr(sensor, 'cameraParams_annot'):
                raise RuntimeError(
                    f"[OceanSim][env {env_idx}] cameraParams_annot not created after sonar_initialize()."
                )
            
            # 验证注释器是否正确附加
            print(f"[Sensor {env_idx}] Checking annotator attachment...")
            if hasattr(sensor, '_render_product_path'):
                print(f"[Sensor {env_idx}] Render product path: {sensor._render_product_path}")
                
                # 检查注释器是否真的附加了
                if hasattr(sensor, 'pointcloud_annot'):
                    try:
                        # 尝试获取注释器信息（不实际获取数据）
                        annotator_info = sensor.pointcloud_annot
                        print(f"[Sensor {env_idx}] Pointcloud annotator: {annotator_info}")
                    except Exception as e:
                        print(f"[Sensor {env_idx}] WARNING: Could not access pointcloud annotator info: {e}")
                else:
                    print(f"[Sensor {env_idx}] ERROR: pointcloud_annot not found!")
            
            # 测试扫描功能
            print(f"[Sensor {env_idx}] Testing initial scan...")
            try:
                import time
                # 等待一个时间步确保注释器完全就绪
                time.sleep(0.1)
                
                test_scan = sensor.scan()
                print(f"[Sensor {env_idx}] Initial scan result: {test_scan}")
                
                if test_scan:
                    print(f"[Sensor {env_idx}] SUCCESS: Initial scan completed, annotator is working!")
                else:
                    print(f"[Sensor {env_idx}] WARNING: Initial scan returned False (may be due to no objects in view)")
            except Exception as scan_error:
                print(f"[Sensor {env_idx}] ERROR: Initial scan failed: {scan_error}")
                print(f"[Sensor {env_idx}] This indicates a potential annotator configuration issue")
            
            self._sensors[env_idx] = sensor
            self._initialized = True
            print(f"[Sensor {env_idx}] Initialization completed successfully")
            
        except Exception as e:
            raise RuntimeError(f"[OceanSim][env {env_idx}] Failed during sonar_initialize pipeline: {e}") from e

    @staticmethod
    def _run_replicator_orchestrator_step() -> None:
        """推进 Replicator SDG，降低 ``Replicator_pointcloud`` 路径未注册时的 KeyError。"""
        try:
            import omni.replicator.core as rep

            orch = getattr(rep, "orchestrator", None)
            if orch is not None:
                orch.step(pause_timeline=False)
        except Exception:
            pass

    def _sonar_scan_with_recovery(self, sensor) -> bool:
        """``sensor.scan()``：KeyError 时先重试一次；若开启 cfg 再试前可 ``rep.orchestrator.step``。"""
        for attempt in range(2):
            try:
                return sensor.scan()
            except KeyError:
                if attempt == 0:
                    continue
                if self.cfg.use_replicator_orchestrator_recovery:
                    self._run_replicator_orchestrator_step()
                try:
                    return sensor.scan()
                except KeyError:
                    return False

    def _sonar_make_and_extract(self, env_idx: int, sensor) -> bool:
        kwargs = dict(
            binning_method=self.cfg.binning_method,
            normalizing_method=self.cfg.normalizing_method,
            attenuation=self.cfg.attenuation,
            gau_noise_param=self.cfg.gau_noise_param,
            ray_noise_param=self.cfg.ray_noise_param,
            intensity_offset=self.cfg.intensity_offset,
            intensity_gain=self.cfg.intensity_gain,
        )
        for attempt in range(2):
            try:
                sensor.make_sonar_data(**kwargs)
                self._extract_sonar_data(env_idx, sensor)
                return True
            except KeyError:
                if attempt == 0:
                    if self.cfg.use_replicator_orchestrator_recovery:
                        self._run_replicator_orchestrator_step()
                    if not self._sonar_scan_with_recovery(sensor):
                        return False
                    continue
                return False

    def reset_envs(self, env_ids: Sequence[int]):
        """重置指定环境的声呐传感器数据
        
        在环境重置时调用，只重置数据缓冲区，不重新初始化传感器
        
        Args:
            env_ids: 需要重置的环境索引列表
        """
        # 重置声纳图像数据
        self._sonar_image[env_ids] = torch.zeros(
            len(env_ids), 
            self._sonar_image.shape[1], 
            self._sonar_image.shape[2], 
            device=self.device
        )
        
        # 重置波束数据
        self._beam_data[env_ids] = torch.zeros(
            len(env_ids), 
            self._beam_data.shape[1], 
            self._beam_data.shape[2], 
            device=self.device
        )
        
        # 重置错误计数
        if hasattr(self, '_annotator_error_count'):
            for env_idx in env_ids:
                if env_idx in self._annotator_error_count:
                    del self._annotator_error_count[env_idx]
        
        # 重置后备数据
        if hasattr(self, '_last_valid_sonar_image') and self._last_valid_sonar_image is not None:
            for env_idx in env_ids:
                if env_idx < self._last_valid_sonar_image.shape[0]:
                    self._last_valid_sonar_image[env_idx] = torch.zeros(
                        self._last_valid_sonar_image.shape[1], 
                        self._last_valid_sonar_image.shape[2], 
                        device=self.device
                    )
        
        if hasattr(self, '_last_valid_beam_data') and self._last_valid_beam_data is not None:
            for env_idx in env_ids:
                if env_idx < self._last_valid_beam_data.shape[0]:
                    self._last_valid_beam_data[env_idx] = torch.zeros(
                        self._last_valid_beam_data.shape[1], 
                        self._last_valid_beam_data.shape[2], 
                        device=self.device
                    )
    
    def update(self, dt: float):
        """更新所有声呐传感器
        
        Args:
            dt: 时间步长
        """
        if not self._initialized:
            return
            
        for env_idx in range(self.num_envs):
            sensor = self._sensors[env_idx]
            if sensor is not None:
                try:
                    scan_success = self._sonar_scan_with_recovery(sensor)
                    if scan_success and self._sonar_make_and_extract(env_idx, sensor):
                        continue

                    if not scan_success:
                        if not hasattr(self, "_annotator_error_count"):
                            self._annotator_error_count = {}
                        self._annotator_error_count[env_idx] = self._annotator_error_count.get(env_idx, 0) + 1
                        self._handle_empty_frame(env_idx)
                        continue

                    if not hasattr(self, "_annotator_error_count"):
                        self._annotator_error_count = {}
                    self._annotator_error_count[env_idx] = self._annotator_error_count.get(env_idx, 0) + 1
                    if self._annotator_error_count[env_idx] == 1:
                        print(
                            f"WARNING: [env {env_idx}] Replicator sonar failed after recovery "
                            f"(e.g. missing SDG node Replicator_pointcloud). Using last valid sonar."
                        )
                    elif self._annotator_error_count[env_idx] % 100 == 0:
                        print(
                            f"INFO: env {env_idx}: {self._annotator_error_count[env_idx]} "
                            "Replicator sonar fallbacks"
                        )
                    self._handle_empty_frame(env_idx)

                except Exception as e:
                    print(f"ERROR: Unexpected error in env {env_idx}: {type(e).__name__}: {e}")
                    self._handle_empty_frame(env_idx)
    
    def _check_annotator_health(self, sensor):
        """检查注释器是否健康
        
        Args:
            sensor: 声呐传感器实例
            
        Returns:
            bool: 注释器是否健康
        """
        try:
            # 检查必要的属性是否存在
            if not hasattr(sensor, 'pointcloud_annot'):
                return False
            if not hasattr(sensor, 'semanticSeg_annot'):
                return False
            if not hasattr(sensor, 'cameraParams_annot'):
                return False
            
            # 尝试访问注释器数据（不获取实际数据）
            import omni.replicator.core as rep
            try:
                # 检查注释器是否在缓存中
                if hasattr(rep, 'AnnotatorCache'):
                    # 这是一个基本检查，不实际获取数据
                    pass
            except:
                return False
                
            return True
            
        except Exception as e:
            return False
    
    def _handle_empty_frame(self, env_idx):
        """处理空帧或错误帧
        
        Args:
            env_idx: 环境索引
        """
        # 保持上一帧的数据，或者使用零数据
        if hasattr(self, '_last_valid_sonar_image') and self._last_valid_sonar_image is not None:
            self._sonar_image[env_idx] = self._last_valid_sonar_image[env_idx].clone()
        else:
            # 使用零数据
            self._sonar_image[env_idx] = torch.zeros(
                self._sonar_image.shape[1], 
                self._sonar_image.shape[2], 
                device=self.device
            )
        
        if hasattr(self, '_last_valid_beam_data') and self._last_valid_beam_data is not None:
            self._beam_data[env_idx] = self._last_valid_beam_data[env_idx].clone()
        else:
            self._beam_data[env_idx] = torch.zeros(
                self._beam_data.shape[1], 
                self._beam_data.shape[2], 
                device=self.device
            )
    
    def debug_info(self):
        """获取声纳传感器调试信息
        
        Returns:
            dict: 包含GPU内存、点云数量等调试信息的字典
        """
        import torch
        debug_info = {
            'num_sensors': len(self._sensors),
            'initialized': self._initialized,
            'active_sensors': sum(1 for s in self._sensors if s is not None),
            'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
            'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0,
        }
        
        if self._initialized and self._sensors[0] is not None:
            sensor = self._sensors[0]
            try:
                labels = sensor.semanticSeg_annot.get_data()['info']['idToLabels']
                debug_info['labels_loaded'] = len(labels)
            except:
                debug_info['labels_loaded'] = 0
            
            try:
                if hasattr(sensor, 'scan_data') and 'pcl' in sensor.scan_data:
                    debug_info['point_cloud_size'] = sensor.scan_data['pcl'].shape[0]
            except:
                debug_info['point_cloud_size'] = 0
        
        return debug_info

    def _extract_sonar_data(self, env_idx: int, sensor):
        """从声呐传感器提取数据
        
        Args:
            env_idx: 环境索引
            sensor: 声呐传感器实例
        """
        if not hasattr(sensor, 'sonar_image'):
            return
            
        sonar_img = sensor.sonar_image.numpy()
        
        if sonar_img is None or sonar_img.size == 0:
            return
            
        # 提取单通道灰度图
        if sonar_img.ndim == 3 and sonar_img.shape[-1] >= 1:
            sonar_img = sonar_img[..., 0]
        
        actual_height, actual_width = sonar_img.shape
        if self._actual_height is None:
            self._actual_height = actual_height
            self._actual_width = actual_width

        sonar_tensor = torch.from_numpy(sonar_img).float()
        # OceanSim outputs may be either [0, 1] floats or [0, 255] image-like values.
        # Normalize only when needed; avoid double-scaling that collapses beam intensity.
        max_val = float(torch.max(sonar_tensor).item()) if sonar_tensor.numel() > 0 else 0.0
        if max_val > 1.5:
            sonar_tensor = sonar_tensor / 255.0
        # 固定到 cfg 尺寸，保证 RL 观测维数稳定（与 Lh2EnvCfg.observation_space 一致）
        if sonar_tensor.shape[0] != self.cfg.height or sonar_tensor.shape[1] != self.cfg.width:
            if not self._resize_notice_logged:
                print(
                    f"Note: OceanSim sonar output [{actual_height}, {actual_width}] -> "
                    f"resized to cfg [{self.cfg.height}, {self.cfg.width}] for policy observations."
                )
                self._resize_notice_logged = True
            sonar_tensor = F.interpolate(
                sonar_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.cfg.height, self.cfg.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        self._sonar_image[env_idx] = sonar_tensor

        # 保存最后的有效数据用于错误恢复
        if self._last_valid_sonar_image is None:
            self._last_valid_sonar_image = torch.zeros(
                self.num_envs,
                self._sonar_image.shape[1],
                self._sonar_image.shape[2],
                device=self.device,
            )
        self._last_valid_sonar_image[env_idx] = sonar_tensor.clone()

        self._extract_beam_data(env_idx, sonar_tensor, sensor)

    def _extract_beam_data(self, env_idx: int, sonar_image: torch.Tensor, sensor):
        """从声呐图像提取波束数据
        
        Args:
            env_idx: 环境索引
            sonar_image: 声呐图像张量
            sensor: 声呐传感器实例
        """
        beam_count = self.cfg.beam_count
        fov = sensor.hori_fov
        width = sonar_image.shape[1]
        height = sonar_image.shape[0]
        
        tensor_device = sonar_image.device
        if (
            self._beam_cache_angles is None
            or self._beam_cache_center_x is None
            or self._beam_cache_width != int(width)
            or self._beam_cache_fov != float(fov)
            or self._beam_cache_angles.device != tensor_device
            or self._beam_cache_angles.shape[0] != beam_count
        ):
            angles = torch.linspace(-fov / 2, fov / 2, beam_count, device=tensor_device)
            center_x = (width / 2 + angles * width / fov).long()
            center_x = torch.clamp(center_x, 0, width - 1)
            self._beam_cache_angles = angles
            self._beam_cache_center_x = center_x
            self._beam_cache_width = int(width)
            self._beam_cache_fov = float(fov)
        else:
            angles = self._beam_cache_angles
            center_x = self._beam_cache_center_x

        beam_data = torch.zeros(beam_count, 3, device=tensor_device)
        beam_data[:, 0] = angles  # 角度

        # 向量化提取每个波束列，替代 Python 循环（降低每步开销）
        beam_cols = sonar_image[:, center_x]  # [H, beam_count]

        topk = max(1, int(getattr(self.cfg, "beam_topk", 1)))
        topk = min(topk, beam_cols.shape[0])
        top_vals, top_idx = torch.topk(beam_cols, k=topk, dim=0, largest=True, sorted=False)
        top_idx = top_idx.float()
        weight_sum = top_vals.sum(dim=0)
        min_weight_sum = float(getattr(self.cfg, "beam_topk_min_weight_sum", 1.0e-4))

        # Top-k weighted center is more robust than argmax under speckle/noise spikes.
        weighted_idx = torch.sum(top_vals * top_idx, dim=0) / torch.clamp(weight_sum, min=1.0e-8)
        argmax_idx = torch.argmax(beam_cols, dim=0).float()
        reliable = weight_sum >= min_weight_sum
        range_idx = torch.where(reliable, weighted_idx, argmax_idx)

        denom = float(max(height - 1, 1))
        beam_data[:, 1] = self.cfg.min_range + (range_idx / denom) * (
            self.cfg.max_range - self.cfg.min_range
        )
        beam_data[:, 2] = torch.max(top_vals, dim=0).values
        
        self._beam_data[env_idx] = beam_data
        
        # 保存最后的有效beam数据用于错误恢复
        if self._last_valid_beam_data is None:
            self._last_valid_beam_data = torch.zeros(
                self.num_envs, 
                self._beam_data.shape[1], 
                self._beam_data.shape[2], 
                device=self.device
            )
        self._last_valid_beam_data[env_idx] = beam_data.clone()

    def get_sonar_image(self) -> torch.Tensor:
        """获取声呐图像
        
        Returns:
            声呐图像张量 [num_envs, height, width]
        """
        return self._sonar_image

    def get_beam_data(self) -> torch.Tensor:
        """获取波束数据
        
        Returns:
            波束数据张量 [num_envs, beam_count, 3]
        """
        return self._beam_data

    def close(self):
        """关闭所有声呐传感器，清理资源"""
        for sensor in self._sensors:
            if sensor is not None and hasattr(sensor, 'close'):
                sensor.close()
        self._sensors = []
        self._initialized = False

    def reset(self):
        """重置声呐传感器"""
        pass