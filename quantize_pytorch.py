# quantize_pytorch.py
import os
import sys
import time
import copy
import argparse
import torch
from transformers import AutoTokenizer

# 导入 torchao 量化模块
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# 导入 MiniMind 模型定义（确保路径正确）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'minimind'))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(args):
    """
    加载模型，返回 model 和 tokenizer。
    不使用 .half()，保留 FP32 精度以用于量化。
    """
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # 2. 构建模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    )
    model = MiniMindForCausalLM(lm_config)
    
    # 3. 构造权重文件路径
    moe_suffix = '_moe' if args.use_moe else ''
    weight_file = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    print(f"  加载权重: {weight_file}")
    
    # 4. 加载权重
    state_dict = torch.load(weight_file, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    
    # 5. 切换到推理模式，保持 FP32
    model.eval()
    model.to(args.device)
    
    # 6. 打印参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {total_params:.2f} M")
    return model, tokenizer


def benchmark_generate(model, tokenizer, prompt, max_new_tokens=50, runs=5, device='cpu'):
    """
    测速函数：使用聊天模板包装 prompt，然后重复生成 runs 次，返回平均耗时（秒）。
    """
    # 关键：使用 apply_chat_template 包装输入（与 eval_llm.py 一致）
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # 如果分词器没有聊天模板（如预训练模型），则使用续写模式
        formatted_prompt = tokenizer.bos_token + prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    # 预热（只生成 5 个 token）
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    
    # 正式测速
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
    avg_time = (time.time() - start) / runs
    return avg_time


def print_model_size(path):
    """打印文件大小（MB）并返回"""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  文件大小: {size_mb:.2f} MB")
    return size_mb


def main():
    parser = argparse.ArgumentParser(description="MiniMind 动态量化测试 (torchao)")
    # 模型加载参数（与 eval_llm.py 保持一致）
    parser.add_argument('--tokenizer_path', default='model', type=str, help="分词器目录 (默认: model)")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀 (如 full_sft, pretrain)")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构 (0=否, 1=是)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用 RoPE 外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    
    # 量化与测试参数
    parser.add_argument('--test_prompt', default='你好，请介绍一下自己', type=str, help="测试提示词")
    parser.add_argument('--max_new_tokens', default=50, type=int, help="生成最大 token 数")
    parser.add_argument('--runs', default=5, type=int, help="测速重复次数")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 1: 加载原始模型（保持 FP32 精度）")
    print("=" * 60)
    model_fp32, tokenizer = load_model(args)
    original_weight_path = f'{args.save_dir}/{args.weight}_{args.hidden_size}{"_moe" if args.use_moe else ""}.pth'
    size_orig = print_model_size(original_weight_path)
    
    # 快速诊断：加载后直接生成一次，验证是否正常（可选）
    print("\n" + "=" * 60)
    print("诊断：加载后直接生成（不量化）")
    print("=" * 60)
    diag_prompt = "中国的首都是"
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        diag_messages = [{"role": "user", "content": diag_prompt}]
        diag_input = tokenizer.apply_chat_template(diag_messages, tokenize=False, add_generation_prompt=True)
    else:
        diag_input = tokenizer.bos_token + diag_prompt
    diag_ids = tokenizer(diag_input, return_tensors='pt').to(args.device)
    with torch.no_grad():
        diag_out = model_fp32.generate(diag_ids['input_ids'], max_new_tokens=10, do_sample=False)
    diag_text = tokenizer.decode(diag_out[0][len(diag_ids['input_ids'][0]):], skip_special_tokens=True)
    print(f"  续写结果: {diag_text}")
    
    print("\n" + "=" * 60)
    print("Step 2: 使用 torchao 进行动态量化 (FP32 → INT8)")
    print("=" * 60)
    model_int8 = copy.deepcopy(model_fp32)
    quantize_(model_int8, Int8DynamicActivationInt8WeightConfig())
    
    # 保存量化后的权重
    int8_path = f'{args.save_dir}/{args.weight}_{args.hidden_size}{"_moe" if args.use_moe else ""}_int8.pth'
    torch.save(model_int8.state_dict(), int8_path)
    print(f"  INT8 权重已保存: {int8_path}")
    size_int8 = print_model_size(int8_path)
    
    print("\n" + "=" * 60)
    print("Step 3: 速度对比 (FP32 vs INT8)")
    print("=" * 60)
    test_prompt = args.test_prompt
    print(f"  测试 Prompt: '{test_prompt}'")
    print("  正在预热并测试 FP32 ...")
    t_fp32 = benchmark_generate(model_fp32, tokenizer, test_prompt, args.max_new_tokens, args.runs, args.device)
    
    print("  正在预热并测试 INT8 ...")
    t_int8 = benchmark_generate(model_int8, tokenizer, test_prompt, args.max_new_tokens, args.runs, args.device)
    
    print(f"\n  FP32 平均耗时: {t_fp32*1000:.2f} ms")
    print(f"  INT8 平均耗时: {t_int8*1000:.2f} ms")
    print(f"  加速比: {t_fp32/t_int8:.2f}x")
    print(f"  体积压缩: {(1 - size_int8/size_orig)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Step 4: 生成效果对比")
    print("=" * 60)
    # 使用聊天模板包装测试 prompt
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": test_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = tokenizer.bos_token + test_prompt
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(args.device)
    
    print("\n[FP32 生成]:")
    with torch.no_grad():
        out_fp32 = model_fp32.generate(inputs['input_ids'], max_new_tokens=30, do_sample=False)
    response_fp32 = tokenizer.decode(out_fp32[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(response_fp32)
    
    print("\n[INT8 生成]:")
    with torch.no_grad():
        out_int8 = model_int8.generate(inputs['input_ids'], max_new_tokens=30, do_sample=False)
    response_int8 = tokenizer.decode(out_int8[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(response_int8)


if __name__ == '__main__':
    main()