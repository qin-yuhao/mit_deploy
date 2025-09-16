import can
import time
import struct

class MotorController:
    """
    Controller for the motor using CAN communication protocol.
    Provides methods to control position, velocity, and torque.
    """
    
    # CAN command constants - 更新特殊指令名称与值
    CMD_ENTER_MOTOR_MODE = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]  # 进入电机模式
    CMD_EXIT_MOTOR_MODE = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]   # 退出电机模式
    CMD_ZERO_RESET = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]        # 零点复位
    
    # 参数范围常量
    P_MIN = -4 * 3.14159  # 位置最小值 (-4π)
    P_MAX = 4 * 3.14159   # 位置最大值 (4π)
    V_MIN = -30.0         # 速度最小值 (-30rad/s)
    V_MAX = 30.0          # 速度最大值 (30rad/s)
    KP_MIN = 0.0          # Kp最小值
    KP_MAX = 500.0        # Kp最大值
    KD_MIN = 0.0          # Kd最小值
    KD_MAX = 5.0          # Kd最大值
    T_MIN = -32.0         # 扭矩最小值 (-18NM)
    T_MAX = 32.0          # 扭矩最大值 (18NM)
    
    def __init__(self, channel='can0', bustype='socketcan', bitrate=1000000, device_ids=[0x01]):
        """
        Initialize the motor controller with CAN bus settings.
        
        Args:
            channel: CAN interface name
            bustype: CAN interface type
            bitrate: CAN bus bitrate (default 1M as per spec)
            device_ids: List of target device IDs
        """
        self.bus = can.Bus(channel=channel, interface=bustype, bitrate=bitrate)
        self.device_ids = device_ids if isinstance(device_ids, list) else [device_ids]
        
        
    def send_command(self, data):
        """
        Send a CAN command to multiple motors with specific device IDs.
        
        Args:
            data: List of bytes to send
        """
        for device_id in self.device_ids:
            msg = can.Message(
                arbitration_id=device_id,
                data=data,
                is_extended_id=False  # 标准帧模式
            )
            try:
                self.bus.send(msg)
                #print(f"发送消息到ID {hex(device_id)}: {[hex(x) for x in data]}")
            except can.CanError as e:
                print(f"发送CAN消息到ID {hex(device_id)}时出错: {e}")
            
    def receive_response(self, timeout=1.0):
        """
        Receive and parse the motor's response.
        
        Args:
            timeout: Time to wait for response in seconds
            
        Returns:
            dict: Contains position, velocity, and current information
        """
        msg = self.bus.recv(timeout)
        if msg is None:
            #print("Timeout: No message received")
            return None
            
        # 解析6字节反馈包
        # 设备ID从数据的第一个字节获取
        device_id = msg.data[0]
        
        # 实际位置 (16位) - 从data[1]和data[2]读取
        position = (msg.data[1] << 8 | msg.data[2])
        
        # 实际速度 (12位) - 从data[3]和部分data[4]读取
        velocity = (msg.data[3] << 4 | (msg.data[4] >> 4) & 0x0F)
        
        # 实际电流/扭矩 (12位) - 从部分data[4]和data[5]读取
        current = ((msg.data[4] & 0x0F) << 8 | msg.data[5])
        
        # 转换为实际物理单位
        position_rad = self._map_value(position, 0, 65535, self.P_MIN, self.P_MAX)
        velocity_rad_s = self._map_value(velocity, 0, 4095, self.V_MIN, self.V_MAX)
        current_amp = self._map_value(current, 0, 4095, -self.T_MAX, self.T_MAX)
        print(f"设备ID {hex(device_id)}: 位置={position_rad:.4f} rad, 速度={velocity_rad_s:.2f} rad/s, 电流={current_amp:.2f} A")
        print(f"原始数据: msg = {msg.data}")
        return {
            'device_id': device_id,
            'position': position_rad,
            'velocity': velocity_rad_s,
            'current': current_amp,
            'raw_position': position,
            'raw_velocity': velocity,
            'raw_current': current
        }
    
    def _map_value(self, value, in_min, in_max, out_min, out_max):
        """
        将一个范围内的值映射到另一个范围
        """
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
    def _scale_to_int(self, value, in_min, in_max, bits):
        """
        将物理值比例缩放为对应的整数表示
        """
        max_value = (1 << bits) - 1
        scaled = int((value - in_min) * max_value / (in_max - in_min))
        return max(0, min(scaled, max_value))  # 确保值在有效范围内
        
    def enter_motor_mode(self):
        """进入电机模式"""
        self.send_command(self.CMD_ENTER_MOTOR_MODE)
        
    def exit_motor_mode(self):
        """退出电机模式"""
        self.send_command(self.CMD_EXIT_MOTOR_MODE)
        
    def zero_reset(self):
        """零点复位"""
        self.send_command(self.CMD_ZERO_RESET)
        
    def set_position(self, position, velocity=0, kp=0, kd=0, torque=0):
        """
        Send position control command to the motor.
        
        Args:
            position: Target position in radians (range: -4π to 4π)
            velocity: Target velocity (range: -30 to 30 rad/s)
            kp: Position control gain (range: 0 to 500)
            kd: Velocity control gain (range: 0 to 5)
            torque: Feed-forward torque (range: -18 to 18 NM)
        """
        # 确保参数在有效范围内
        position = max(self.P_MIN, min(position, self.P_MAX))
        velocity = max(self.V_MIN, min(velocity, self.V_MAX))
        kp = max(self.KP_MIN, min(kp, self.KP_MAX))
        kd = max(self.KD_MIN, min(kd, self.KD_MAX))
        torque = max(self.T_MIN, min(torque, self.T_MAX))
        
        # 转换为整数表示
        pos_int = self._scale_to_int(position, self.P_MIN, self.P_MAX, 16)
        vel_int = self._scale_to_int(velocity, self.V_MIN, self.V_MAX, 12) 
        kp_int = self._scale_to_int(kp, self.KP_MIN, self.KP_MAX, 12)
        kd_int = self._scale_to_int(kd, self.KD_MIN, self.KD_MAX, 12)
        torque_int = self._scale_to_int(torque, self.T_MIN, self.T_MAX, 12)
        
        # 分解为字节
        pos_high = (pos_int >> 8) & 0xFF
        pos_low = pos_int & 0xFF
        
        vel_high = (vel_int >> 4) & 0xFF
        vel_low = (vel_int & 0x0F) << 4
        
        kp_high = (kp_int >> 8) & 0x0F
        kp_low = kp_int & 0xFF
        
        kd_high = (kd_int >> 4) & 0xFF
        kd_low = (kd_int & 0x0F) << 4
        
        torque_high = (torque_int >> 8) & 0x0F
        torque_low = torque_int & 0xFF
        
        # 构造命令
        data = [
            pos_high, pos_low,
            vel_high, vel_low | (kp_high & 0x0F),
            kp_low, 
            kd_high, kd_low | (torque_high & 0x0F),
            torque_low
        ]
        
        self.send_command(data)
        for resu in range(len(self.device_ids)):
            result = self.receive_response()
            if result is None:
                print(f"设备ID {hex(self.device_ids[resu])} 没有响应")
            else:
                pass
                #print(f"设备ID {hex(result['device_id'])} 响应: 位置={result['position']:.2f} rad, 速度={result['velocity']:.2f} rad/s, 电流={result['current']:.2f} A")

        return result
        
    def set_angle_degrees(self, degrees, velocity=0, kp=0, kd=0, torque=0):
        """
        Set the motor position in degrees.
        
        Args:
            degrees: Target position in degrees
            velocity: Target velocity
            kp: Position control gain
            kd: Velocity control gain
            torque: Feed-forward torque
        """
        # 将角度转换为弧度
        radians = degrees * (3.14159 / 180.0)
        return self.set_position(position=radians, velocity=velocity, kp=kp, kd=kd, torque=torque)
    
    def close(self):
        """Close the CAN bus connection."""
        self.bus.shutdown()

def main():
    """Example usage of the motor controller."""
    try:
        # Initialize the motor controller

        motor = MotorController(channel='can1', bustype='socketcan', bitrate=1000000, device_ids=[0x01,0x02,0x03,0x04,0x05,0x06]) #,, ,,,  ,0x01,0x02,0x03,0x04,0x05, ,0x02,0x03,0x04,0x05,0x06
        print(f"电机控制器已初始化，控制的设备ID: {[hex(id) for id in motor.device_ids]}")
        
        

        
        # 零点复位
        # motor.zero_reset()
        # time.sleep(0.5)


        # 退出电机模式(复位)
        motor.exit_motor_mode()
        time.sleep(0.5)

        motor.receive_response()  # 接收退出模式的反馈
        # 进入电机模式
        motor.enter_motor_mode()
        time.sleep(0.5)
        motor.receive_response()  # 接收进入模式的反馈
        
        
        
        # 设置不同位置
        angle = 0

        frames_count = 1
        start_time = time.time()
        for i in range(frames_count):
            #print(f"第 {i+1} 次循环")
            #print(f"移动到 {angle} 度")
            # 使用适当的增益，注意kd范围是0-5
            response = motor.set_angle_degrees(angle, kp=0.0, kd=1, torque=0.0)
            #print(f"电机响应: {response}")
            #time.sleep(0.0001)  # 等待电机达到位置
        
        
        end_time = time.time()
        #计算fps
        fps = frames_count / (end_time - start_time)
        print(f"总帧数: {frames_count}, 耗时: {end_time - start_time:.2f}秒, FPS: {fps:.2f}")


        # 退出电机模式并关闭连接
        print("退出电机模式并关闭连接...")
        motor.exit_motor_mode()
        print("已退出电机模式")
        
        #读取多余消息
        print("读取多余消息...")
        for _ in range(100):
            info = motor.receive_response(timeout=0.01)  # 确保接收所有剩余消息
            if info is not None:
                pass
                #print(f"多余消息: {info}")
        print("电机控制器已关闭")
        motor.close()
        print("关闭CAN连接")
        
    except Exception as e:
        print(f"错误: {e}")
        # 发生异常时也尝试安全关闭
        try:
            motor.exit_motor_mode()
            motor.close()
        except:
            pass

if __name__ == "__main__":
    main()