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
                +1, -1, +1,  # 左前腿
                -1, +1, -1,  # 右前腿  
                +1, -1, +1,  # 左后腿
                +1, +1, -1   # 右后腿
            ]
        
        if self.zero_positions is None:
            self.zero_positions = [
                0.0,  0.0,   0.0,  # 左前腿 (FL): hip, thigh, calf
                0.0,  0.0,   0.0,  # 右前腿 (FR): hip, thigh, calf
                0.0,  0.0,   0.0,  # 左后腿 (RL): hip, thigh, calf
                0.0,  0.0,   0.0,   # 右后腿 (RR): hip, thigh, calf
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
            self.pose = [
                -0.0, -1.5, 0,    # FL_hip_joint, FL_thigh_joint, FL_calf_joint
                -0.0, -1.5, 0,    # FR_hip_joint, FR_thigh_joint, FR_calf_joint
                -0.0,  1.5, -0,   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
                -0.0,  1.5, -0    # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            ]
        if self.kp is None:
            self.kp = [8] * 12
        if self.kd is None:
            self.kd = [1] * 12


@dataclass
class StandUpConfig(PoseConfig):
    """站立姿态配置 (模型顺序)"""
    pose: List[float] = None
    kp: List[float] = None
    kd: List[float] = None
    
    def __post_init__(self):
        if self.pose is None:
            self.pose = [
                0.0, 0.5, 1.1,    # FL_hip_joint, FL_thigh_joint, FL_calf_joint
                -0.0, 0.5, 1.1,    # FR_hip_joint, FR_thigh_joint, FR_calf_joint
                0.0,  -0.5, -1.1,   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
                -0.0,  -0.5, -1.1    # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            ]
        if self.kp is None:
            self.kp = [12] * 12
        if self.kd is None:
            self.kd = [0.5] * 12


@dataclass
class ScaleConfig:
    """缩放配置 (模型顺序)"""
    action: List[float] = None
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    ang_vel: float = 0.25

 
    def __post_init__(self):
        if self.action is None:
            self.action = [0.25 for _ in range(12)]  # 默认动作缩放为0.0
            #hip_decimation = 0.5
            
            # 通过全局常量设置髋关节的缩放
            # for i, name in enumerate(MODEL_JOINT_NAMES):
            #     if "hip" in name:
            #         self.action[i] = hip_decimation * self.action[i]
                
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
    model_path: str = "/home/cat/mit_deploy/policy.onnx"

    # 缩放配置
    scale: ScaleConfig = None
    
    def __post_init__(self):
        if self.pose is None:
            self.pose = [
                0.0, 0.5, 1.1,    # FL_hip_joint, FL_thigh_joint, FL_calf_joint
                -0.0, 0.5, 1.1,    # FR_hip_joint, FR_thigh_joint, FR_calf_joint
                0.0,  -0.5, -1.1,   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
                -0.0,  -0.5, -1.1    # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            ]
        
        if self.kp is None:
            self.kp = [0] * 12
        
        if self.kd is None:
            self.kd = [0.5] * 12
        
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
