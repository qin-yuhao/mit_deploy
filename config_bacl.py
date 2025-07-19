#!/usr/bin/env python3
"""
机器人配置文件 - Python版本
基于 devq.yaml 转换而来，提供更好的类型检查和IDE支持
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# 全局常量：模型关节名称（模型顺序）
MODEL_JOINT_NAMES = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",        # 髋关节
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", # 大腿关节
    "FL_calf", "FR_calf", "RL_calf", "RR_calf"     # 小腿关节
]


@dataclass
class RobotConfig:
    """机器人基础配置"""
    dt: float = 0.02  # 控制周期 (200Hz)


@dataclass
class ActuatorConfig:
    """执行器配置"""
    # 执行器方向相对于关节 (参考devq.yaml的actuator_directions)
    directions: List[int] = None
    
    # 关节零点位置 (参考devq.yaml的joint_zero_pos)
    zero_positions: List[float] = None
    
    def __post_init__(self):
        if self.directions is None:
            self.directions = [
                +1, +1, +1,  # 左前腿 (FL): hip, thigh, calf
                -1, -1, -1,  # 右前腿 (FR): hip, thigh, calf
                -1, -1, -1,  # 左后腿 (RL): hip, thigh, calf
                +1, +1, +1   # 右后腿 (RR): hip, thigh, calf
            ]
            # self.directions = [
            #     +1, -1, -1, #fl_hip, fl_thigh, fl_calf
            #     +1, -1, +1, #fr_hip, fr_thigh, fr_calf
            #     +1, -1, +1, #rl_hip, rl_thigh, rl_calf
            #     -1, -1, +1  #rr_hip, rr_thigh, rr_calf
            # ]
        
        if self.zero_positions is None:
            self.zero_positions = [
                0.0044,  -1.4755,   -2.723859,  # 左前腿 (FL): hip, thigh, calf
                0.0190,   1.6032,    2.709659,  # 右前腿 (FR): hip, thigh, calf
                -0.0151,  1.7202,  -2.724459,  # 左后腿 (RL): hip, thigh, calf
                -0.0232, -1.5967,    2.708859   # 右后腿 (RR): hip, thigh, calf
            ]


@dataclass
class IMUConfig:
    """IMU配置"""
    port: str = "/dev/ttyACM0"
    baudrate: int = 921600


@dataclass
class PoseConfig:
    """姿态配置"""
    pose: List[float]
    kp: List[float]
    kd: List[float]
    duration: float = 6.0


@dataclass
class LieDownConfig(PoseConfig):
    """趴下姿态配置 (模型顺序)"""
    pose: List[float] = None
    kp: List[float] = None
    kd: List[float] = None
    
    def __post_init__(self):
        if self.pose is None:
            # 模型顺序: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf
            self.pose = [
                -0.6, -0.6, -0.6, -0.6,    # 髋关节: FL, FR, RL, RR
                -1.17, -1.17, 1.17, 1.17,  # 大腿关节: FL, FR, RL, RR
                2.7, 2.7, -2.7, -2.7       # 小腿关节: FL, FR, RL, RR
            ]
        if self.kp is None:
            self.kp = [0] * 12
        if self.kd is None:
            self.kd = [0.0] * 12


@dataclass
class StandUpConfig(PoseConfig):
    """站立姿态配置 (模型顺序)"""
    pose: List[float] = None
    kp: List[float] = None
    kd: List[float] = None
    
    def __post_init__(self):
        if self.pose is None:
            # 模型顺序: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf
            self.pose = [
                -0.0, -0.0, -0.0, -0.0,     # 髋关节: FL, FR, RL, RR
                -0.8, -0.8, 1.0, 1.0,       # 大腿关节: FL, FR, RL, RR
                1.5, 1.5, -1.5, -1.5        # 小腿关节: FL, FR, RL, RR
            ]
        if self.kp is None:
            self.kp = [0] * 12
        if self.kd is None:
            self.kd = [0] * 12


@dataclass
class ScaleConfig:
    """缩放配置 (模型顺序)"""
    action: List[float] = None
    dof_pos: float = 1.0
    dof_vel: float = 1.0
    
    def __post_init__(self):
        if self.action is None:
            self.action = [0.21 for _ in range(12)]  # 默认动作缩放为0.0
            hip_decimation = 0.5
            
            # 通过全局常量设置髋关节的缩放
            for i, name in enumerate(MODEL_JOINT_NAMES):
                if "hip" in name:
                    self.action[i] = hip_decimation * self.action[i]
                
            print(f"Action scale set to: {self.action}")
            


@dataclass
class RLModelConfig:
    """强化学习模型配置 (模型顺序)"""
    # 默认姿态 (按模型输出顺序)
    pose: List[float] = None
    kp: List[float] = None
    kd: List[float] = None
    
    decimation: int = 1
    #model_path: str = "/home/cat/mit_dog_ctl/policy719.onnx"
    model_path: str = "/home/cat/deploy/policy.onnx"

    # 缩放配置
    scale: ScaleConfig = None
    
    def __post_init__(self):
        if self.pose is None:
            self.pose = [
                -0.0, -0.0, -0.0, -0.0,  # 髋关节: FL, FR, RL, RR
                -0.8, -0.8, 1.0, 1.0,    # 大腿关节: FL, FR, RL, RR
                1.5, 1.5, -1.5, -1.5,    # 小腿关节: FL, FR, RL, RR
            ]
        
        if self.kp is None:
            self.kp = [0] * 12
        
        if self.kd is None:
            self.kd = [0] * 12
        
        if self.scale is None:
            self.scale = ScaleConfig()


@dataclass
class JointNamesConfig:
    """关节名称配置 (模型顺序)"""
    # 模型关节名称 (按模型输出顺序)
    model_joint_names: List[str] = None
    
    def __post_init__(self):
        if self.model_joint_names is None:
            self.model_joint_names = MODEL_JOINT_NAMES.copy()
