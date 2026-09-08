# 导入环境和相关库
import os
import sys
import torch
import onnx
import onnxruntime
import argparse
import numpy as np
from transformers import AutoTokenizer
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType

# 导入Minimind模型定义（确保路径正确）
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
# 加载模型相关配置
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# 加载模型
def load_model(args):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # 构建模型配置
    llm_config = MiniMindConfig(
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layers,
        use_moe = bool(args.use_moe),
        inference_rope_scaling = args.inference_rope_scaling
    )
    model = MiniMindForCausalLM(llm_config)

    # 构造权重文件路径
    moe_suffix = "_moe" if args.use_moe else ''
    weight_file = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    print(f"加载权重为{weight_file}")

    # 加载权重
    state_dice = torch.load(weight_file,map_location='cpu')
    model.load_state_dict(state_dice, strict=True)

    # 切换模式
    model.half().eval()
    model.to(args.device)

    # 打印参数量
    total_params = sum(p.numel()for p in model.parameters()) / 1e6
    print(f"总参数量为：{total_params:.2f} M")
    return model, tokenizer, weight_file

def print_onnx_states(onnx_path):
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"当前模型大小为：{onnx_size} M")

def test_model(args, model, tokenizer):

    # 构建对话
    conversation = []
    if not args.use_auto_prompt:
        prompt = input('请输入测试提示词：')
    else:
        prompt = "你是谁"
    conversation.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(conversation, tokenize = False,add_generation_prompt=True, open_thinking=bool(0))
    inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
    input_ids = inputs['input_ids']
    generate_output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False
    )
    response = tokenizer.decode(generate_output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"模型生成：{response}")

def onnx_fp32(args, model):
    # ---- 强制 freqs_cos 初始化（可选） ----
    if hasattr(model, 'model') and hasattr(model.model, 'freqs_cos'):
        model.model.freqs_cos = torch.ones_like(model.model.freqs_cos)
        model.model.freqs_sin = torch.zeros_like(model.model.freqs_sin)

    dummy_input = torch.randint(0, 1000, (1, 10), dtype=torch.long, device='cpu')
    
    # 获取示例输出（只取 logits）
    with torch.no_grad():
        example_output = model(dummy_input).logits

    onnx_fp32_path = 'out/minimind_model.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'seq_len'}, 'logits': {0: 'batch_size', 1: 'seq_len'}},
        opset_version=14,
        verbose=True,  
        do_constant_folding=True,
        dynamo = False,
        training=torch.onnx.TrainingMode.EVAL
    )
    print("导出onnx成功")
    print_onnx_states(onnx_fp32_path)
    return onnx_fp32_path
    

def onnx_model_test(onnx_path):
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("Model is structurally valid.") # 模型结构有效[reference:24]
    except onnx.checker.ValidationError as e:
        print(f"Model validation failed: {e}") 
    return onnx_model

def onnx_generate(args, onnx_model, tokenizer):
    session = onnxruntime.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
    output_names = [session.get_outputs()[0].name]
    
    # ---- 自动探测并补齐所有输入节点 ----
    input_feed = {}
    for inp in session.get_inputs():
        name = inp.name
        shape = inp.shape
        if name == 'input_ids':
            continue  # 后面再填充
        # 对额外输入（如 position_ids / onnx::Gather_2）自动构造 arange 数据
        # 典型 shape 为 [batch_size, seq_len] 或 [seq_len]
        seq_len = 1  # 初始占位，后面根据实际 input_ids 更新
        if len(shape) == 2:
            arr = np.arange(seq_len).reshape(1, seq_len).astype(np.int64)
        elif len(shape) == 1:
            arr = np.arange(seq_len).astype(np.int64)
        else:
            arr = np.zeros([d if isinstance(d, int) else 1 for d in shape], dtype=np.int64)
        input_feed[name] = arr
    
    conversation = []
    if not args.use_auto_prompt:
        prompt = input('请输入测试提示词：')
    else:
        prompt = "中国的首都是"
    conversation.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(conversation, tokenize = False,add_generation_prompt=True, open_thinking=bool(0))
    inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
    input_ids = inputs['input_ids']
    response = []
    # 模拟自回归
    for _ in range(args.max_new_tokens):
        # 更新 input_ids
        input_feed['input_ids'] = input_ids.cpu().numpy().astype(np.int64)
        # 同步更新所有额外输入的序列长度
        seq_len = input_ids.shape[1]
        for inp in session.get_inputs():
            name = inp.name
            if name == 'input_ids':
                continue
            shape = inp.shape
            if len(shape) == 2:
                input_feed[name] = np.arange(seq_len).reshape(1, seq_len).astype(np.int64)
            elif len(shape) == 1:
                input_feed[name] = np.arange(seq_len).astype(np.int64)
        
        outputs = session.run(output_names, input_feed)
        logits_tensor = outputs[0]
        
        # ---- 兼容 ONNX 模型可能输出的多种维度 ----
        if logits_tensor.ndim == 1:
            # 输出为 (vocab_size,)
            next_token = int(np.argmax(logits_tensor))
        elif logits_tensor.ndim == 2:
            # 输出为 (seq_len, vocab_size) 或 (1, vocab_size)，取最后一个位置
            next_token = int(np.argmax(logits_tensor[-1]))
        else:
            # 标准三维输出 (batch, seq, vocab)，取最后一个 token 的 logits
            logits = logits_tensor[0, -1, :]
            next_token = int(np.argmax(logits, axis=-1))
        
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=input_ids.device)], dim=1)
        response.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    responses = tokenizer.decode(response, skip_special_tokens=True)
    print(f"onnx模型回复为：{responses}")

def onnx__dynamic_int8(args, onnx_path, tokenizer):
    output_onnx = f"model_dynamic_int8.onnx"
    quantize_dynamic(
        model_input = onnx_path,
        model_output = output_onnx,
        weight_type = QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"] 
    )
    print("onnx量化q8成功")
    print_onnx_states(output_onnx)
    onnx_model_test(output_onnx)
    onnx_generate(args, output_onnx, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Minimind 动态量化命令行参数")
    parser.add_argument('--tokenizer_path', default='model', type=str, help="分词器目录 (默认: model)")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀 (如 full_sft, pretrain)")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构 (0=否, 1=是)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用 RoPE 外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    
    # 量化与测试参数
    parser.add_argument('--use_auto_prompt', default= 1, type=int, help="测试提示词")
    parser.add_argument('--max_new_tokens', default=50, type=int, help="生成最大 token 数")

    args = parser.parse_args()

    model, tokenizer, model_path = load_model(args)
    print(f"基座模型为：{model_path}")

    print("="*50)
    print("检测基座模型是否正常")
    test_model(args,model,tokenizer)

    print('='*50)
    print('检测导出的onnx模型是否正常')
    onnx_path = onnx_fp32(args, model)
    onnx_model = onnx_model_test(onnx_path)
    onnx_generate(args, onnx_path,tokenizer)

    print("="*50)
    print("检测qint8的onnx模型是否正常")
    onnx__dynamic_int8(args,onnx_path, tokenizer)

if __name__ == "__main__":
    main()