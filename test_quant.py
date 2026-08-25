# 手写一个量化与反量化函数
import torch


def quantize_dequantize(fp32_tensor,num_bits=8):
    # 模拟量化-反量化全过程（非对称量化）

    # 确定取值范围
    q_max, q_min = 2**num_bits-1, 0

    # 计算缩放参数和零点
    R_max, R_min = fp32_tensor.max(), fp32_tensor.min() 
    S = (R_max-R_min)/(q_max-q_min)
    Z = q_min-torch.round(R_min/S)
    Z = torch.clamp(Z, q_min, q_max)  # 截断，控制范围大小

    # 映射公式
    t = torch.round(fp32_tensor//S + Z)
    X = torch.clamp(t, q_min, q_max)  # 截断

    # 反量化
    q_tensor = (X- Z)*S

    # 测试
    print(f"原始数据: {fp32_tensor}")
    print(f"量化后整数: {X}")
    print(f"反量化后: {q_tensor}")
    print(f"误差: {torch.abs(fp32_tensor - q_tensor)}")
    
    return q_tensor



test_data = torch.tensor([-1.2, 0.5, 2.3, 5.1, -0.8])
quantize_dequantize(test_data)