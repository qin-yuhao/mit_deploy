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
    'FL_hip', 'FL_thigh', 'FL_calf',
    'FR_hip', 'FR_thigh', 'FR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
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
        
        if self.zero_positions is None:
            # self.zero_positions = [
            #     0.0044,  -1.4755,   -2.723859,  # 左前腿 (FL): hip, thigh, calf
            #     0.0190,   1.6032,    2.709659,  # 右前腿 (FR): hip, thigh, calf
            #     -0.0151,  1.7202,  -2.724459,  # 左后腿 (RL): hip, thigh, calf
            #     -0.0232, -1.5967,    2.708859   # 右后腿 (RR): hip, thigh, calf
            # ]
            # self.zero_positions = [
            #     0.0514, -1.4990, -2.7934,
            #     0.0238, 1.4082,  2.7907 - 2.141,
            #     0.0882, 1.6891,  -2.7742,
            #     -0.0974,-1.6584,  2.7497
            # ]

            self.zero_positions = [
                0.0858, -1.5690, -2.8003,
                0.0307, 1.4867,  0.6536,
                -0.0521, 1.6190,  -2.7960,
                0.0644,-1.6319,  2.7918
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
                -0.6, -1.17, 2.7,    # FL_hip_joint, FL_thigh_joint, FL_calf_joint
                -0.6, -1.17, 2.7,    # FR_hip_joint, FR_thigh_joint, FR_calf_joint
                -0.6,  1.17, -2.7,   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
                -0.6,  1.17, -2.7    # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            ]
        if self.kp is None:
            self.kp = [4] * 12
        if self.kd is None:
            self.kd = [0.5] * 12


@dataclass
class StandUpConfig(PoseConfig):
    """站立姿态配置 (模型顺序)"""
    pose: List[float] = None
    kp: List[float] = None
    kd: List[float] = None
    
    def __post_init__(self):
        if self.pose is None:
            self.pose = [
                # -0.2, -0.8, 1.2,    # FL_hip_joint, FL_thigh_joint, FL_calf_joint
                # -0.1, -0.85, 1.2,    # FR_hip_joint, FR_thigh_joint, FR_calf_joint
                # -0.0,  0.7, -1.4,   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
                # -0.0,  0.65, -1.4    # RR_hip_joint, RR_thigh_joint, RR_calf_joint
                -0.1, -0.6, 1.2,    # FL_hip_joint, FL_thigh_joint, FL_calf_joint
                -0.1, -0.6, 1.2,    # FR_hip_joint, FR_thigh_joint, FR_calf_joint
                -0.1,  0.7, -1.4,   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
                -0.1,  0.65, -1.4    # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            ]
        if self.kp is None:
            self.kp = [25] * 12
        if self.kd is None:
            self.kd = [0.5] * 12


@dataclass
class ScaleConfig:
    """缩放配置 (模型顺序)"""
    action: List[float] = None
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    ang_vel: float = 0.25
    command_lin: float = 1.0
    command_ang: float = 0.5

 
    def __post_init__(self):
        if self.action is None:
            self.action = [0.25 for _ in range(12)]  # 默认动作缩放为0.0
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
    model_path: str = "/home/cat/deploy/model2.onnx"

    # 缩放配置
    scale: ScaleConfig = None
    
    def __post_init__(self):
        if self.pose is None:
            self.pose = [
                -0.1, -0.6, 1.2,    # FL_hip_joint, FL_thigh_joint, FL_calf_joint
                -0.1, -0.6, 1.2,    # FR_hip_joint, FR_thigh_joint, FR_calf_joint
                -0.1,  0.7, -1.4,   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
                -0.1,  0.7, -1.4    # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            ]
        
        if self.kp is None:
            self.kp = [25]* 12
        
        if self.kd is None:
            self.kd = [0.8] * 12
        
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
