#!/usr/bin/env python3
"""
ONNX模型推理速度测试脚本
用于测试四足机器人控制模型的推理性能
"""

import sys
import os
import time
import numpy as np
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append('/home/cat')
sys.path.append('/home/cat/deploy')

from config import RLModelConfig
import onnxruntime as ort


class ONNXInferenceTester:
    """ONNX模型推理测试器"""
    
    def __init__(self, model_path=None):
        # 初始化配置
        self.rl_config = RLModelConfig()
        self.model_path = model_path or self.rl_config.model_path
        
        # 观测维度：4 command + 6 imu + 36 joints = 46
        # 注意：实际代码中去掉了lin_acc，所以是45维
        self.obs_dim = 45
        self.history_len = 10  # 历史观测长度
        
        # 初始化观测缓冲区
        self.obs_history_buf = np.zeros((self.history_len, self.obs_dim))
        
        # 初始化ONNX模型
        self.init_onnx_model()
        
    def init_onnx_model(self):
        """初始化ONNX模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_path}")
            
        print(f"加载ONNX模型: {self.model_path}")
        
        # 创建会话选项
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 加载模型
        self.onnx_session = ort.InferenceSession(
            self.model_path, 
            sess_options=session_options
        )
        
        # 获取输入输出信息
        self.input_names = [input.name for input in self.onnx_session.get_inputs()]
        self.output_names = [output.name for output in self.onnx_session.get_outputs()]
        
        print(f"✓ ONNX模型加载成功")
        print(f"  - 输入名称: {self.input_names}")
        print(f"  - 输出名称: {self.output_names}")
        print(f"  - 输入形状: {[input.shape for input in self.onnx_session.get_inputs()]}")
        
    def generate_random_observation(self):
        """生成随机观测数据"""
        # 生成随机观测数据
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        
        # 更新历史观测缓冲区
        self.obs_history_buf = np.concatenate([
            self.obs_history_buf[1:, :],  # 移除最旧的观测
            obs[np.newaxis, :]            # 添加新的观测
        ], axis=0)
        
        return obs
        
    def load_observation_data(self, obs_file, obs_his_file):
        """加载观测数据，与RKNN保持一致"""
        # 加载观测数据
        print(f"Loading observation data from {obs_file}")
        obs_data = np.load(obs_file)
        print(f"Observation data shape: {obs_data.shape}")
        
        # 加载历史观测数据
        print(f"Loading history observation data from {obs_his_file}")
        obs_his_data = np.load(obs_his_file)
        print(f"History observation data shape: {obs_his_data.shape}")
        
        return obs_data, obs_his_data
        
    def run_inference_from_files(self, obs_file, obs_his_file):
        """从文件运行模型推理，与RKNN保持一致"""
        # 加载数据
        obs_data, obs_his_data = self.load_observation_data(obs_file, obs_his_file)
        
        if len(obs_data) > 0 and len(obs_his_data) > 0:
            obs_prop = obs_data[0]
            obs_history = obs_his_data[0]
            
            print(f"Current observation shape: {obs_prop.shape}")
            print(f"History observation shape: {obs_history.shape}")
            
            # 预处理输入数据，与RKNN保持一致
            obs_prop_input = obs_prop.reshape(1, -1).astype(np.float16)
            obs_hist_full = obs_history.reshape(1, self.history_len, -1).astype(np.float16)
            
            # 构建输入字典，与RKNN保持一致的输入顺序
            inputs = {}
            if len(self.input_names) >= 2:
                inputs[self.input_names[0]] = obs_prop_input
                inputs[self.input_names[1]] = obs_hist_full
            else:
                # 如果只有一个输入，使用历史观测作为输入
                inputs[self.input_names[0]] = obs_hist_full
                
            # 运行推理
            outputs = self.onnx_session.run(None, inputs)
            action = outputs[0].squeeze()
            
            return action
        else:
            print("No data available for inference")
            return None
        
    def run_inference(self, observation):
        """运行模型推理"""
        # 双输入模型，分别处理当前观测和历史观测
        # obs_prop：当前时刻的观测数据
        obs_prop_input = observation.reshape(1, -1).astype(np.float32)
        
        # 生成随机历史观测数据
        obs_hist_data = np.random.randn(self.history_len, self.obs_dim).astype(np.float32)
        obs_hist_full = obs_hist_data.reshape(1, self.history_len, -1).astype(np.float32)
        
        # 构建输入字典
        inputs = {}
        if len(self.input_names) >= 2:
            inputs[self.input_names[0]] = obs_prop_input
            inputs[self.input_names[1]] = obs_hist_full
        else:
            # 如果只有一个输入，使用历史观测作为输入
            inputs[self.input_names[0]] = obs_hist_full
            
        # 运行推理
        outputs = self.onnx_session.run(None, inputs)
        action = outputs[0].squeeze()
        
        return action
        
    def benchmark_inference(self, num_iterations=1000, warmup_iterations=100):
        """基准测试推理速度"""
        print(f"开始推理速度测试...")
        print(f"  - 预热迭代次数: {warmup_iterations}")
        print(f"  - 测试迭代次数: {num_iterations}")
        
        # 预热阶段
        print("预热中...")
        for i in range(warmup_iterations):
            obs = self.generate_random_observation()
            _ = self.run_inference(obs)
            
        # 测试阶段
        print("正式测试中...")
        inference_times = []
        
        for i in range(num_iterations):
            obs = self.generate_random_observation()
            
            start_time = time.perf_counter()
            action = self.run_inference(obs)
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)
            
            if (i + 1) % 100 == 0:
                print(f"已完成 {i + 1}/{num_iterations} 次推理")
        
        # 统计结果
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_time = np.std(inference_times)
        
        # 计算百分位数
        p50 = np.percentile(inference_times, 50)
        p90 = np.percentile(inference_times, 90)
        p95 = np.percentile(inference_times, 95)
        p99 = np.percentile(inference_times, 99)
        
        print("\n" + "="*50)
        print("推理性能测试结果:")
        print("="*50)
        print(f"平均推理时间: {avg_time:.3f} ms")
        print(f"最小推理时间: {min_time:.3f} ms")
        print(f"最大推理时间: {max_time:.3f} ms")
        print(f"推理时间标准差: {std_time:.3f} ms")
        print(f"推理时间中位数: {p50:.3f} ms")
        print(f"90%分位数: {p90:.3f} ms")
        print(f"95%分位数: {p95:.3f} ms")
        print(f"99%分位数: {p99:.3f} ms")
        print(f"平均推理频率: {1000/avg_time:.1f} Hz")
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'frequency': 1000/avg_time
        }
        
    def test_single_inference(self):
        """测试单次推理"""
        print("执行单次推理测试...")
        
        # 生成观测数据
        obs = self.generate_random_observation()
        
        # 执行推理
        start_time = time.perf_counter()
        action = self.run_inference(obs)
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        print(f"观测维度: {obs.shape}")
        print(f"动作维度: {action.shape}")
        print(f"推理时间: {inference_time:.3f} ms")
        print(f"动作示例 (前3维): {action[:3]}")


def main():
    parser = argparse.ArgumentParser(description='ONNX模型推理速度测试')
    parser.add_argument('--model', type=str, help='ONNX模型路径')
    parser.add_argument('--obs_file', type=str, help='观测数据文件路径 (observations_*.npy)', default="/home/cat/observations_1755874526.npy")
    parser.add_argument('--obs_his_file', type=str, help='历史观测数据文件路径 (observations_his*.npy)',default="/home/cat/observations_his1755874526.npy")
    parser.add_argument('--iterations', type=int, default=1000, help='测试迭代次数 (默认: 1000)')
    parser.add_argument('--warmup', type=int, default=100, help='预热迭代次数 (默认: 100)')
    parser.add_argument('--single', action='store_true', help='只执行单次推理测试')
    parser.add_argument('--from_files', action='store_true', help='从文件加载数据进行推理')
    
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = ONNXInferenceTester(model_path=args.model)
        
        action = tester.run_inference_from_files(args.obs_file, args.obs_his_file)
        if action is not None:
            print(f"动作维度: {action.shape}")
            print(f"动作值: {action}")
        else:
            # 基准测试
            tester.benchmark_inference(
                num_iterations=args.iterations,
                warmup_iterations=args.warmup
            )
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("✓ 测试完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())