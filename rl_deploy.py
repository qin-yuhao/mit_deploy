#!/usr/bin/env python3
"""
简单的强化学习部署代码
基于ONNX Runtime的四足机器人控制
"""

import sys
import os
import time
import numpy as np
import argparse
from pathlib import Path
from config import *

# 创建配置实例
RobotConfig = RobotConfig()
ActuatorConfig = ActuatorConfig()
IMUConfig = IMUConfig()
RLModelConfig = RLModelConfig()
JointNamesConfig = JointNamesConfig()
LieDownConfig = LieDownConfig()
StandUpConfig = StandUpConfig()
# 添加构建目录到Python路径

import socket
import struct
import threading

# 修复导入路径
sys.path.append('/home/cat')
from motor_ctl.joint import JointController
sys.path.append('/home/cat/dm_imu/build')
import dm_imu_py
import onnxruntime as ort




class RLController:
    """强化学习控制器"""
    
    def __init__(self):
        self.iteration = 0
        self.start_time = time.time()
        
        # 状态变量
        self.last_action = np.zeros(12)  # 12个关节
        self.cmd_vel = np.zeros(3)  # [vx, vy, wz]
        self.target_height = 0.2  # 默认目标高度
        
        # 模式控制
        self.current_mode = 0  # 0=PASSIVE, 1=LIE_DOWN, 2=STAND_UP, 3=RL_MODEL, 4=SOFT_STOP
        self.mode_changed = False  # 模式变化标志
        
        # 观测相关
        self.obs_dim = 46  # 4 command + 6 imu(去掉lin_acc) + 36 joints
        
        # 初始化命令接收器
        self.init_command_receiver()
        
        # 初始化ONNX Runtime
        self.init_onnx_model()
        
        # 初始化硬件接口
        self.init_hardware()
        
    def init_command_receiver(self):
        """初始化UDP命令接收器"""
        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_sock.bind(('127.0.0.1', 5526))
        self.command_sock.settimeout(0.1)  # 非阻塞接收
        
        # 命令接收线程
        self.command_thread = threading.Thread(target=self._command_receiver_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        print("✓ UDP命令接收器初始化成功 (端口: 5526)")
    
    def _command_receiver_loop(self):
        """命令接收循环"""
        while True:
            try:
                data, addr = self.command_sock.recvfrom(1024)
                if len(data) == 17:  # 1 byte mode + 4 floats (4 bytes each)
                    # 解析命令: mode (1 byte) + vx, vy, wz, target_height (4 floats)
                    mode, vx, vy, wz, target_height = struct.unpack('=Bffff', data)
                    
                    # 检查模式变化
                    if mode != self.current_mode:
                        self.current_mode = mode
                        self.mode_changed = True
                        print(f"模式切换: {self.get_mode_name(mode)}")
                    
                    # 更新命令
                    self.cmd_vel = np.array([vx, vy, wz])
                    self.target_height = target_height
                    
                    #print(f"收到命令: mode={mode}, vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}, height={target_height:.3f}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"命令接收错误: {e}")
                time.sleep(0.1)
        
    def init_onnx_model(self):
        """初始化ONNX模型"""
        if not os.path.exists(RLModelConfig.model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {RLModelConfig.model_path}")
            
        # 创建会话选项
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 加载模型
        self.onnx_session = ort.InferenceSession(
            RLModelConfig.model_path, 
            sess_options=session_options
        )
        
        # 获取输入输出信息
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name
        
        print(f"✓ ONNX模型加载成功")
        print(f"  - 输入名称: {self.input_name}")
        print(f"  - 输出名称: {self.output_name}")
        
    def init_hardware(self):
        """初始化硬件接口"""
        # 初始化关节控制器
        print("初始化关节控制器...")
        self.joint_controller = JointController(JointNamesConfig.model_joint_names,directions=ActuatorConfig.directions)
        
        # 初始化IMU
        print("初始化IMU...")
        try:
            self.imu = dm_imu_py.DmImu(IMUConfig.port, IMUConfig.baudrate)
            print(f"✓ IMU初始化成功: {IMUConfig.port}")
        except Exception as e:
            print(f"⚠️ IMU初始化失败: {e}")
            self.imu = None
            
        print("✓ 硬件接口初始化完成")
        
    def get_mode_name(self, mode):
        """获取模式名称"""
        mode_names = {
            0: "PASSIVE",
            1: "LIE_DOWN", 
            2: "STAND_UP",
            3: "RL_MODEL",
            4: "SOFT_STOP"
        }
        return mode_names.get(mode, f"UNKNOWN({mode})")
        
    def handle_mode_change(self):
        """处理模式变化"""
        if not self.mode_changed:
            return
            
        self.mode_changed = False
        print(f"执行模式切换: {self.get_mode_name(self.current_mode)}")
        
        if self.current_mode == 0:  # PASSIVE
            self.execute_passive_mode()
        elif self.current_mode == 1:  # LIE_DOWN
            self.execute_lie_down_mode()
        elif self.current_mode == 2:  # STAND_UP
            self.execute_stand_up_mode()
        elif self.current_mode == 3:  # RL_MODEL
            self.execute_rl_model_mode()
        elif self.current_mode == 4:  # SOFT_STOP
            self.execute_soft_stop_mode()
    
    def execute_passive_mode(self):
        """执行被动模式 - 关闭所有力矩"""
        print("进入被动模式...")
        self.joint_controller.set_joint_commands(
            positions=[0.0] * 12,
            kp_gains=[0.0] * 12,
            kd_gains=[3.0] * 12
        )
        
    def execute_lie_down_mode(self):
        """执行趴下模式"""
        print("进入趴下模式...")
        self.move_to_pose(LieDownConfig.pose, LieDownConfig.kp, LieDownConfig.kd, duration=6.0)
        
    def execute_stand_up_mode(self):
        """执行站立模式"""
        print("进入站立模式...")
        self.move_to_pose(StandUpConfig.pose, StandUpConfig.kp, StandUpConfig.kd, duration=6.0)
        
    def execute_rl_model_mode(self):
        """执行强化学习模式"""
        print("进入强化学习模式...")
        # 先移动到默认姿态
        self.move_to_default_pose(duration=2.0)
        
    def execute_soft_stop_mode(self):
        """执行软停止模式 - 保持阻尼"""
        print("进入软停止模式...")
        self.joint_controller.set_joint_commands(
            positions=[0.0] * 12,
            kp_gains=[0.0] * 12,
            kd_gains=[2.0] * 12  # 保持阻尼
        )
        
    def move_to_pose(self, target_pose, kp_gains, kd_gains, duration=3.0):
        """移动到指定姿态"""
        print(f"移动到目标姿态，耗时 {duration:.1f} 秒...")
        
        # 获取当前位置
        current_states = self.joint_controller.get_joint_states()
        current_pos = np.array(current_states['positions'])
        
        steps = int(duration / RobotConfig.dt)
        for i in range(steps):
            alpha = i / steps
            target_pos = current_pos * (1 - alpha) + np.array(target_pose) * alpha
            
            self.joint_controller.set_joint_commands(
                positions=target_pos.tolist(),
                kp_gains=kp_gains,
                kd_gains=kd_gains
            )
            
            time.sleep(RobotConfig.dt)
            
        print("✓ 已到达目标姿态")
        
    def get_gravity_orientation(self, quaternion):
        """使用四元数计算重力投影"""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
        
    def get_observation(self):
        """获取观测数据"""
        #发送上一次关节控制

        # 获取关节状态
        joint_states = self.joint_controller.get_joint_states()
        joint_pos = np.array(joint_states['positions'])
        joint_vel = np.array(joint_states['velocities'])
        
        # 打印当前关节位置
        if self.iteration % 200 == 0:  # 每200步打印一次，避免输出过多
            print(f"[{self.iteration}] 当前关节位置: {joint_pos.round(3)}")

        # 获取IMU数据
        imu_data = self.imu.get_latest_data()
        ang_vel = np.array([imu_data.gyrox, imu_data.gyroy, imu_data.gyroz])
        # 去掉线性加速度观测
        # lin_acc = np.array([imu_data.accx, imu_data.accy, imu_data.accz])
        
        # 计算四元数（从欧拉角）
        roll_rad = imu_data.roll * np.pi / 180.0
        pitch_rad = imu_data.pitch * np.pi / 180.0
        yaw_rad = imu_data.yaw * np.pi / 180.0
        
        # 将欧拉角转换为四元数
        cy = np.cos(yaw_rad * 0.5)
        sy = np.sin(yaw_rad * 0.5)
        cp = np.cos(pitch_rad * 0.5)
        sp = np.sin(pitch_rad * 0.5)
        cr = np.cos(roll_rad * 0.5)
        sr = np.sin(roll_rad * 0.5)
        
        quaternion = np.array([
            cy * cp * cr + sy * sp * sr,  # w
            cy * cp * sr - sy * sp * cr,  # x
            sy * cp * sr + cy * sp * cr,  # y
            sy * cp * cr - cy * sp * sr   # z
        ])
        
        # 使用四元数计算重力投影
        gravity_vec = self.get_gravity_orientation(quaternion)
        
        # 计算相对关节位置
        relative_pos = (joint_pos - np.array(RLModelConfig.pose)) * RLModelConfig.scale.dof_pos
        scaled_vel = joint_vel * RLModelConfig.scale.dof_vel
        ang_vel = ang_vel * RLModelConfig.scale.ang_vel
        cmd = self.cmd_vel
        #最后一维度乘以0.5
        cmd[2] *= 0.5  # wz缩放为0.5
        # 组装观测向量 - 按照C++代码的顺序（去掉线性加速度）
        obs = np.concatenate([
            # 命令信息 (4维)
            cmd,              # 3: vx, vy, wz
            [self.target_height],      # 1: target_height
            
            # IMU数据 (6维) - 去掉线性加速度
            ang_vel,                   # 3: 角速度 (gyro_x, gyro_y, gyro_z)
            # lin_acc,                 # 3: 线加速度 (acc_x, acc_y, acc_z) - 已去掉
            gravity_vec,               # 3: 重力投影 (proj_x, proj_y, proj_z)
            
            # 关节数据 (36维)
            relative_pos,              # 12: 相对关节位置
            scaled_vel,                # 12: 关节速度
            self.last_action           # 12: 上次动作
        ])
        
        return obs
        
    def run_inference(self, observation):
        """运行模型推理"""
        # 准备输入
        input_data = observation.reshape(1, -1).astype(np.float32)
        
        # 运行推理
        outputs = self.onnx_session.run([self.output_name], {self.input_name: input_data})
        action = outputs[0].squeeze()
        
        return action
        
    def apply_action(self, action):
        """应用动作到关节"""
        # 计算目标关节位置
        # 更新上次动作
        self.last_action = action.copy()
        target_positions = np.array(RLModelConfig.pose) + action * np.array(RLModelConfig.scale.action)
        
        # 发送关节命令
        self.joint_controller.set_joint_commands(
            positions=target_positions.tolist(),
            kp_gains=RLModelConfig.kp,
            kd_gains=RLModelConfig.kd
        )
        
        
        
    def set_command_velocity(self, vx=0.0, vy=0.0, wz=0.0):
        """设置命令速度"""
        self.cmd_vel = np.array([vx, vy, wz])
        # 限制命令速度
        max_cmd_vel = 2.0  # 最大命令速度 (m/s 和 rad/s)
        self.cmd_vel = np.clip(self.cmd_vel, -max_cmd_vel, max_cmd_vel)
        
    def run_step(self):
        """运行一个控制步骤"""
        # 处理模式变化
        self.handle_mode_change()
        
        # 只有在RL_MODEL模式下才执行强化学习控制
        if self.current_mode == 3:  # RL_MODEL
            # 每隔decimation步进行一次推理
            if self.iteration % RLModelConfig.decimation == 0:
                # 开始计时
                start_time = time.time()
                
                # 获取观测
                self.apply_action(self.last_action)  # 应用上次动作
                obs = self.get_observation()
                # 运行推理
                action = self.run_inference(obs)
                
                # 应用动作
                self.apply_action(action)
                
                # 计算延迟
                end_time = time.time()
                delay_ms = (end_time - start_time) * 1000  # 转换为毫秒
                
                if self.iteration % 100 == 0:  # 每100步打印一次
                    # 打印关节之前的观测数据
                    print(f"[{self.iteration}] 观测前10维: {obs[:10].round(3)}")
                    print(f"[{self.iteration}] 动作: {action.round(3)}")
                    print(f"[{self.iteration}] 处理延迟: {delay_ms:.2f}ms")
        
        self.iteration += 1
        
    def move_to_default_pose(self, duration=3.0):
        """移动到默认姿态"""
        print(f"移动到默认姿态，耗时 {duration:.1f} 秒...")
        
        # 获取当前位置
        current_states = self.joint_controller.get_joint_states()
        current_pos = np.array(current_states['positions'])
        
        steps = int(duration / RobotConfig.dt)
        for i in range(steps):
            alpha = i / steps
            target_pos = current_pos * (1 - alpha) + np.array(RLModelConfig.pose) * alpha
            
            self.joint_controller.set_joint_commands(
                positions=target_pos.tolist(),
                kp_gains=RLModelConfig.kp,
                kd_gains=RLModelConfig.kd
            )
            
            time.sleep(RobotConfig.dt)
            
        print("✓ 已到达默认姿态")
        
    def stop(self):
        """停止控制器"""
        print("停止控制器...")
        # 设置零力矩
        self.joint_controller.set_joint_commands(
            positions=[0.0] * 12,
            kp_gains=[0.0] * 12,
            kd_gains=[3.0] * 12  # 保持阻尼
        )
        
        # 停止关节控制器
        self.joint_controller.stop()
        print("✓ 控制器已停止")


def main():
    parser = argparse.ArgumentParser(description='四足机器人强化学习部署')
    parser.add_argument('--vx', type=float, default=0.0, help='前进速度 (m/s)')
    parser.add_argument('--vy', type=float, default=0.0, help='侧向速度 (m/s)')
    parser.add_argument('--wz', type=float, default=0.0, help='转向速度 (rad/s)')
    args = parser.parse_args()
    
    try:
        # 创建控制器
        controller = RLController()
        
        # 不移动到默认姿态，直接进入被动模式
        print("初始化完成，进入被动模式...")
        controller.execute_passive_mode()
        
        # 设置命令速度
        controller.set_command_velocity(args.vx, args.vy, args.wz)
        
        print(f"开始运行强化学习控制")
        print(f"初始命令速度: vx={args.vx}, vy={args.vy}, wz={args.wz}")
        print("当前模式: PASSIVE (被动模式)")
        print("程序将持续运行，按 Ctrl+C 停止...")
        
        # 主控制循环 - 无限循环
        try:
            while True:
                step_start = time.time()
                
                # 运行一个控制步骤
                controller.run_step()
                
                # 控制循环时间
                elapsed = time.time() - step_start
                sleep_time = max(0, RobotConfig.dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n收到停止信号...")
            
        # 停止控制器
        controller.stop()
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("✓ 程序正常结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
