#!/usr/bin/env python3
"""
ONNX和RKNN模型对比测试脚本
用于比较两种模型的推理速度和输出结果
"""

import numpy as np
import time
import os

# 尝试导入ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("警告: 未安装ONNX Runtime，将跳过ONNX模型测试")
    ONNX_AVAILABLE = False

# 尝试导入RKNN Lite
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    print("警告: 未安装RKNN Lite，将跳过RKNN模型测试")
    RKNN_AVAILABLE = False

# 模型路径
ONNX_MODEL_PATH = '/home/cat/mit_deploy/policy.onnx'
RKNN_MODEL_PATH = '/home/cat/mit_deploy/policy.rknn'

class ONNXModel:
    """ONNX模型包装类"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()
    
    def load_model(self):
        """加载ONNX模型"""
        print(f"正在加载ONNX模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_path}")
        
        # 创建会话选项
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 加载模型
        self.session = ort.InferenceSession(
            self.model_path, 
            sess_options=session_options
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✓ ONNX模型加载成功")
        print(f"  - 输入名称: {self.input_name}")
        print(f"  - 输出名称: {self.output_name}")
        print(f"  - 输入形状: {self.session.get_inputs()[0].shape}")
        print(f"  - 输出形状: {self.session.get_outputs()[0].shape}")
    
    def inference(self, input_data):
        """执行推理"""
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]

class RKNNModel:
    """RKNN模型包装类"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.rknn = None
        self.load_model()
    
    def load_model(self):
        """加载RKNN模型"""
        print(f"正在加载RKNN模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RKNN模型文件不存在: {self.model_path}")
        
        self.rknn = RKNNLite()
        
        # 加载模型
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError('Load RKNN model failed')
        
        # 初始化运行时环境
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError('Init runtime environment failed')
        
        print(f"✓ RKNN模型加载成功")
    
    def inference(self, input_data):
        """执行推理"""
        outputs = self.rknn.inference(inputs=[input_data])
        return outputs[0]
    
    def release(self):
        """释放资源"""
        if self.rknn:
            self.rknn.release()

def test_inference_performance(model, model_name, iterations=1000):
    """测试推理性能"""
    print(f"\n开始{model_name}性能测试 ({iterations}次推理)...")
    
    # 预热运行
    test_input = np.random.rand(1, 45).astype(np.float32)
    for _ in range(10):
        _ = model.inference(test_input)
    
    # 实际性能测试
    start_time = time.time()
    for i in range(iterations):
        # 使用随机输入数据进行测试
        test_input = np.random.rand(1, 45).astype(np.float32)
        output = model.inference(test_input)
        
        if i % 100 == 0:
            print(f"  已完成 {i}/{iterations} 次推理")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / iterations * 1000  # 转换为毫秒
    
    print(f"✓ {model_name}性能测试完成")
    print(f"  - 总耗时: {total_time:.4f} 秒")
    print(f"  - 平均耗时: {avg_time:.4f} 毫秒")
    print(f"  - FPS: {iterations/total_time:.2f}")
    
    return avg_time

def compare_with_sample_data(model, model_name):
    """使用固定输入数据进行推理，便于结果对比"""
    print(f"\n开始{model_name}固定输入推理测试...")
    
    # 使用固定的输入数据（全1）
    test_input = np.ones((1, 45), dtype=np.float32)
    output = model.inference(test_input)
    
    print(f"  输入数据 (前10维): {test_input[0, :10]}")
    print(f"  输出数据 (前10维): {output[0, :10]}")
    print(f"  输出形状: {output.shape}")
    
    return output

def compare_models_output(onnx_output, rknn_output, tolerance=1e-5):
    """比较两个模型的输出结果"""
    print("\n开始模型输出对比...")
    
    # 计算差异
    diff = np.abs(onnx_output - rknn_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  - 最大差异: {max_diff:.6f}")
    print(f"  - 平均差异: {mean_diff:.6f}")
    
    if max_diff < tolerance:
        print("  ✓ 两个模型输出基本一致")
        return True
    else:
        print(f"  ⚠ 两个模型输出存在较大差异 (>{tolerance})")
        return False

def main():
    print("ONNX和RKNN模型对比测试工具")
    print("=" * 50)
    
    onnx_model = None
    rknn_model = None
    results = {}
    
    try:
        # 测试ONNX模型
        if ONNX_AVAILABLE and os.path.exists(ONNX_MODEL_PATH):
            try:
                onnx_model = ONNXModel(ONNX_MODEL_PATH)
                onnx_output = compare_with_sample_data(onnx_model, "ONNX")
                onnx_avg_time = test_inference_performance(onnx_model, "ONNX", iterations=1000)
                results['onnx_time'] = onnx_avg_time
                results['onnx_output'] = onnx_output
            except Exception as e:
                print(f"❌ ONNX模型测试失败: {e}")
        else:
            print("跳过ONNX模型测试")
        
        # 测试RKNN模型
        if RKNN_AVAILABLE and os.path.exists(RKNN_MODEL_PATH):
            try:
                rknn_model = RKNNModel(RKNN_MODEL_PATH)
                rknn_output = compare_with_sample_data(rknn_model, "RKNN")
                rknn_avg_time = test_inference_performance(rknn_model, "RKNN", iterations=1000)
                results['rknn_time'] = rknn_avg_time
                results['rknn_output'] = rknn_output
            except Exception as e:
                print(f"❌ RKNN模型测试失败: {e}")
        else:
            print("跳过RKNN模型测试")
        
        # 结果对比
        print("\n" + "=" * 50)
        print("测试总结:")
        if 'onnx_time' in results and 'rknn_time' in results:
            print(f"  - ONNX平均推理时间: {results['onnx_time']:.4f} ms")
            print(f"  - RKNN平均推理时间: {results['rknn_time']:.4f} ms")
            
            # 速度比较
            if results['onnx_time'] < results['rknn_time']:
                speedup = results['rknn_time'] / results['onnx_time']
                print(f"  - ONNX比RKNN快 {speedup:.2f} 倍")
            else:
                speedup = results['onnx_time'] / results['rknn_time']
                print(f"  - RKNN比ONNX快 {speedup:.2f} 倍")
            
            # 输出一致性比较
            compare_models_output(results['onnx_output'], results['rknn_output'])
        elif 'onnx_time' in results:
            print(f"  - ONNX平均推理时间: {results['onnx_time']:.4f} ms")
            print("  - RKNN模型未测试")
        elif 'rknn_time' in results:
            print(f"  - RKNN平均推理时间: {results['rknn_time']:.4f} ms")
            print("  - ONNX模型未测试")
        else:
            print("  - 两个模型均未成功测试")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 释放资源
        if rknn_model:
            rknn_model.release()
    
    print("\n✓ 对比测试完成")
    return 0

if __name__ == "__main__":
    exit(main())