#!/usr/bin/env python3
"""
简单的RKNN模型推理脚本
读取已保存的npy文件中的观测数据并进行推理
"""

import numpy as np
from rknnlite.api import RKNNLite
import os
import argparse
import time

# RKNN模型文件 (根据你的实际模型文件名修改)
RKNN_MODEL = '/home/cat/deploy/model.rknn'

def load_rknn_model(model_path):
    """加载RKNN模型"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
        
    rknn_lite = RKNNLite()
    
    # Load RKNN model
    print('--> Loading RKNN model')
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print('Load RKNN model failed')
        return None
    print('done')
    
    return rknn_lite

def init_runtime(rknn_lite):
    """初始化运行时环境"""
    print('--> Initializing runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        return False
    print('done')
    return True

def run_inference(rknn_lite, obs_prop, obs_history):
    """运行推理"""
    print('--> Running model')
    
    # 预处理输入数据
    obs_prop_input = obs_prop.reshape(1, -1).astype(np.float32)
    obs_hist_full = obs_history.reshape(1, 10, -1).astype(np.float32)  # 假设历史长度为10
    
    # 构建输入列表
    inputs = [obs_prop_input, obs_hist_full]
    
    # 运行推理
    outputs = rknn_lite.inference(inputs=inputs)
    
    print('done')
    return outputs

def run_inference_multiple_times(rknn_lite, obs_prop, obs_history, iterations=100):
    """运行多次推理并计算平均耗时"""
    print(f'--> Running inference {iterations} times')
    
    # 预处理输入数据
    obs_prop_input = obs_prop.reshape(1, -1).astype(np.float32)
    obs_hist_full = obs_history.reshape(1, 10, -1).astype(np.float32)  # 假设历史长度为10
    
    # 构建输入列表
    inputs = [obs_prop_input, obs_hist_full]
    
    # 预热运行
    print('Warming up...')
    for _ in range(10):
        rknn_lite.inference(inputs=inputs)
    
    # 多次推理测试
    times = []
    for i in range(iterations):
        start_time = time.time()
        outputs = rknn_lite.inference(inputs=inputs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # 计算统计信息
    avg_time = np.mean(times) * 1000  # 转换为毫秒
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    std_time = np.std(times) * 1000
    
    print(f'Inference statistics over {iterations} runs:')
    print(f'  Average time: {avg_time:.2f} ms')
    print(f'  Minimum time: {min_time:.2f} ms')
    print(f'  Maximum time: {max_time:.2f} ms')
    print(f'  Std deviation: {std_time:.2f} ms')
    
    return outputs

def main():
    parser = argparse.ArgumentParser(description='Simple RKNN Inference for Observation Data')
    parser.add_argument('--obs_file', type=str, default="/home/cat/observations_1755874526.npy",
                        help='观测数据文件路径 (observations_*.npy)')
    parser.add_argument('--obs_his_file', type=str, default="/home/cat/observations_his1755874526.npy",
                        help='历史观测数据文件路径 (observations_his*.npy)')
    parser.add_argument('--model', type=str, default=RKNN_MODEL,
                        help='RKNN模型文件路径')
    args = parser.parse_args()
    
    # 加载RKNN模型
    rknn_lite = load_rknn_model(args.model)
    if rknn_lite is None:
        exit(-1)
    
    # 初始化运行时环境
    if not init_runtime(rknn_lite):
        rknn_lite.release()
        exit(-1)
    
    # 加载观测数据
    print(f"Loading observation data from {args.obs_file}")
    obs_data = np.load(args.obs_file)
    print(f"Observation data shape: {obs_data.shape}")
    
    # 加载历史观测数据
    print(f"Loading history observation data from {args.obs_his_file}")
    obs_his_data = np.load(args.obs_his_file)
    print(f"History observation data shape: {obs_his_data.shape}")
    
    # 取第一组数据进行推理
    if len(obs_data) > 0 and len(obs_his_data) > 0:
        obs_prop = obs_data[0]
        obs_history = obs_his_data[0]
        
        print(f"Current observation shape: {obs_prop.shape}")
        print(f"History observation shape: {obs_history.shape}")
        
        # 运行100次推理并计算平均耗时
        outputs = run_inference_multiple_times(rknn_lite, obs_prop, obs_history, 100)
        
        # 输出结果
        print(f"Inference outputs: {outputs}")
        if outputs:
            action = outputs[0]
            print(f"Action shape: {action.shape}")
            print(f"Action values: {action}")
    else:
        print("No data available for inference")
    
    # 释放资源
    rknn_lite.release()
    print("RKNN inference completed")

if __name__ == '__main__':
    main()