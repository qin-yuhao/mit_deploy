#!/usr/bin/env python3
"""
零位校准程序
在趴下上电位置读取当前电机位置，计算新的zero_positions配置
"""

import sys
import time
import numpy as np

# 添加路径
sys.path.append('/home/cat')
from motor_ctl.joint import JointController
from config import ActuatorConfig

def main():
    print("=" * 60)
    print("零位校准程序")
    print("=" * 60)
    print("请确保机器人已经手动摆放到趴下上电位置：")
    print("FL: hip=-0.6283, thigh=-1.0647, calf=2.7227")
    print("FR: hip=-0.6283, thigh=-1.0647, calf=2.7227")
    print("RL: hip=-0.6283, thigh=1.0647,  calf=-2.7053")
    print("RR: hip=-0.6283, thigh=1.0647,  calf=-2.7053")
    print("=" * 60)
    
    # 确认用户准备好了
    input("按回车键继续校准...")
    
    # 目标关节位置（趴下上电位置，按base_joint_names顺序）
    target_joint_positions = [
        -0.6283, -1.0647, 2.7227,  # FL: hip, thigh, calf
        -0.6283, -1.0647, 2.7227,  # FR: hip, thigh, calf  
        -0.6283, 1.0647, -2.7053,  # RL: hip, thigh, calf
        -0.6283, 1.0647, -2.7053   # RR: hip, thigh, calf
    ]
    
    # 创建ActuatorConfig以获取默认的directions
    actuator_config = ActuatorConfig()
    
    # 创建关节控制器，使用config中的directions
    controller = JointController(
        directions=actuator_config.directions,
        zero_positions=[0.0] * 12  # 临时零位，用于读取当前位置
    )
    
    try:
        # 启动控制线程
        controller.start_control_thread(rate=100)
        
        print("等待系统稳定...")
        time.sleep(2)
        
        # 读取当前电机位置（采集多次求平均）
        print("读取电机位置 (采集10次求平均)...")
        motor_positions_samples = []
        
        for i in range(10):
            # 获取原始电机状态
            motor_states = controller.controller.get_all_motor_states()
            motor_positions = [state.position for state in motor_states]
            motor_positions_samples.append(motor_positions)
            print(f"采样 {i+1}/10: {[f'{pos:.4f}' for pos in motor_positions]}")
            time.sleep(0.1)
        
        # 计算平均位置
        avg_motor_positions = np.mean(motor_positions_samples, axis=0)
        
        print("\n" + "=" * 60)
        print("校准结果")
        print("=" * 60)
        print(f"{'关节名称':<12} {'目标关节位置':<12} {'当前电机位置':<12} {'方向':<6} {'新零位':<12}")
        print("-" * 60)
        
        new_zero_positions = []
        
        print("\n调试信息 - 转换公式验证:")
        print("公式: motor_pos = (joint_pos * direction) + zero_pos")
        print("推导: zero_pos = motor_pos - (joint_pos * direction)")
        print()
        
        for i in range(12):
            joint_name = controller.base_joint_names[i]
            target_joint = target_joint_positions[i]
            current_motor = avg_motor_positions[i]
            direction = actuator_config.directions[i]
            
            # 计算新的零位
            # 公式: motor_pos = (joint_pos * direction) + zero_pos
            # 推导: zero_pos = motor_pos - (joint_pos * direction)
            new_zero = current_motor - (target_joint * direction)
            new_zero_positions.append(new_zero)
            
            # 调试验证：用新零位反推关节位置
            check_joint = (current_motor - new_zero) / direction
            
            print(f"{joint_name:<12} {target_joint:>8.4f}     {current_motor:>8.4f}     {direction:>3d}   {new_zero:>8.4f}   (验证:{check_joint:>7.4f})")
        
        print("\n注意: '验证'列应该等于'目标关节位置'列")
        
        # 输出Python配置格式
        print("\n" + "=" * 60)
        print("新的zero_positions配置 (复制到config.py):")
        print("=" * 60)
        print("self.zero_positions = [")
        for i in range(0, 12, 3):
            fl_fr_rl_rr = ['FL', 'FR', 'RL', 'RR'][i//3]
            print(f"    {new_zero_positions[i]:.4f}, {new_zero_positions[i+1]:.4f}, {new_zero_positions[i+2]:.4f},  # {fl_fr_rl_rr}")
        print("]")
        
        # 检查方向配置是否正确
        print("\n" + "=" * 60)
        print("方向配置检查:")
        print("=" * 60)
        expected_directions = [
            [1, 1, 1],    # FL: hip, thigh, calf
            [-1, -1, -1], # FR: hip, thigh, calf  
            [-1, -1, -1], # RL: hip, thigh, calf
            [1, 1, 1]     # RR: hip, thigh, calf
        ]
        
        for leg in range(4):
            leg_names = ['FL', 'FR', 'RL', 'RR']
            actual = [actuator_config.directions[leg*3 + j] for j in range(3)]
            expected = expected_directions[leg]
            match = actual == expected
            print(f"{leg_names[leg]}: 实际={actual}, 期望={expected}, {'✓' if match else '❌'}")
        
        if not all(
            [actuator_config.directions[leg*3 + j] for j in range(3)] == expected_directions[leg] 
            for leg in range(4)
        ):
            print("\n⚠️  警告: 方向配置可能不正确，这会影响零位计算！")
        
        # 关闭当前控制器
        print("\n" + "=" * 60)
        print("验证校准结果...")
        print("=" * 60)
        print("正在关闭当前控制器...")
        controller.stop()
        time.sleep(1)
        
        # 用新零位创建新的控制器
        print("用新零位重新初始化控制器...")
        new_controller = JointController(
            directions=actuator_config.directions,
            zero_positions=new_zero_positions
        )
        
        # 启动新控制器
        new_controller.start_control_thread(rate=100)
        print("等待系统稳定...")
        time.sleep(2)
        
        # 读取关节位置进行验证
        print("读取关节位置进行验证...")
        joint_positions_samples = []
        
        for i in range(5):
            # 获取原始电机状态
            motor_states = new_controller.controller.get_all_motor_states()
            motor_positions = [state.position for state in motor_states]
            
            # 手动转换为关节位置
            joint_positions = []
            for j in range(12):
                joint_pos = (motor_positions[j] - new_zero_positions[j]) / actuator_config.directions[j]
                joint_positions.append(joint_pos)
            
            joint_positions_samples.append(joint_positions)
            print(f"验证采样 {i+1}/5: {[f'{pos:.4f}' for pos in joint_positions]}")
            time.sleep(0.1)
        
        # 计算平均关节位置
        avg_joint_positions = np.mean(joint_positions_samples, axis=0)
        
        print("\n" + "=" * 60)
        print("最终验证结果:")
        print("=" * 60)
        print("关节名称      目标关节位置    实际关节位置      误差      状态")
        print("----------------------------------------------------------")
        
        max_error = 0.0
        all_ok = True
        
        for i in range(12):
            joint_name = new_controller.base_joint_names[i]
            target = target_joint_positions[i]
            actual = avg_joint_positions[i]
            error = actual - target
            abs_error = abs(error)
            
            if abs_error > max_error:
                max_error = abs_error
            
            status = "✓" if abs_error < 0.02 else "❌"  # 允许±0.02弧度误差
            if abs_error >= 0.02:
                all_ok = False
            
            print(f"{status} {joint_name:<12} {target:>8.4f}     {actual:>8.4f}     {error:>+8.4f}     {'OK' if abs_error < 0.02 else 'ERROR'}")
        
        print("\n" + "=" * 60)
        print("校准总结:")
        print("=" * 60)
        print(f"最大误差: {max_error:.4f} 弧度")
        print(f"校准结果: {'✅ 成功' if all_ok else '❌ 失败'}")
        
        if all_ok:
            print("🎉 校准成功！可以将上面的zero_positions配置复制到config.py中使用。")
        else:
            print("⚠️  校准可能存在问题，请检查:")
            print("   1. 机器人是否准确摆放到趴下位置")
            print("   2. 电机是否正常连接和上电")
            print("   3. CAN总线通信是否正常")
        
        # 关闭新控制器
        new_controller.stop()
        # 将controller设为None，避免finally块中重复关闭
        controller = None
            
    except Exception as e:
        print(f"❌ 校准失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保控制器被关闭
        if controller is not None:
            controller.stop()

if __name__ == "__main__":
    main()
