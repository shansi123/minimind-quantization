import os
import sys
import json
import torch
import math
import argparse
from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig, GPTQQuantizer, AutoGPTQForCausalLM

# 导入Minimind模型定义（确保路径正确）
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
# 加载模型相关配置
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling = args.inference_rope_scaling
    )

    model = MiniMindForCausalLM(lm_config)

    use_moe = f"_moe" if args.use_moe else ""
    weight_path = f"{args.save_dir}/{args.weight}_{args.hidden_size}{use_moe}.pth"

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.half().eval().to(args.device)

    total_parameters = sum(p.numel() for p in model.parameters()) / 1e-6
    print(f"参数总数为：{total_parameters} M")

    return model, tokenizer, weight_path

def dataest_process(data_path,tokenizer):

    with open(data_path, "r", encoding="utf-8")as f:
        datas = []
        for i,j in enumerate(f):
            if i >= 128:
                break
            if not j:
                continue
            j = j.strip()
            data = json.loads(j)
            conversations = data.get('conversations', [])
            if not conversations:
                continue
            formatted_text = tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False
            )
            datas.append(formatted_text)
        

        return datas

def GPTQ_achieve(args, weight_path, model, tokenizer, datas):

    gptq_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=0.01,
        desc_act=bool(args.desc_act),
        sym=bool(args.sym)
    )

    example = [tokenizer(text, return_tensors="pt", max_length=512, truncation=True) for text in datas]

    quantizer = GPTQQuantizer(gptq_config)
    quantized_model = quantizer.quantize(
        model,
        examples=example
    )

    quantized_model.to("cpu")

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
    generate_output = quantized_model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False
    )
    response = tokenizer.decode(generate_output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"模型生成：{response}")

    quantized_model.save_quantized(f"{args.save_dir}/quantized_{args.bits}bit")

def main():
    parser = argparse.ArgumentParser(description="GPTQ量化")

    parser.add_argument('--tokenizer_path', default='model', type=str, help="分词器目录 (默认: model)")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀 (如 full_sft, pretrain)")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构 (0=否, 1=是)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用 RoPE 外推")
    parser.add_argument('--device', default='cpu' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    
    # 量化与测试参数
    parser.add_argument('--group_size', default= 16, type=int, help="分组大小")
    parser.add_argument('--bits', default=4, type=int, help="量化为q几")
    parser.add_argument('--desc_act', default=0, type=int, help="是否使用按列激活顺序")
    parser.add_argument('--use_auto_prompt', default=1, type=int, help="是否使用默认提示词")
    parser.add_argument('--max_new_tokens', default=50, type=int, help="最大生成token数")
    parser.add_argument('--data_path', default=r"D:\minimind-quantization\dataset\lora_medical.jsonl", type=str, help="数据路径")
    parser.add_argument('--sym', default=1, type=int, help="是否使用对称量化")

    args = parser.parse_args()

    model, tokenizer, weight_path = load_model(args)
    datas = dataest_process(args.data_path,tokenizer)
    GPTQ_achieve(args, weight_path, model, tokenizer, datas)

if __name__ == "__main__":
    main()