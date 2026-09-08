import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer
from optimum.gptq import GPTQQuantizer  # 修改：使用 optimum.gptq

# 导入Minimind模型定义（确保路径正确）
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
# 加载模型相关配置
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,  # 修改：修复拼写错误 num_hidden_size → num_hidden_layers
        use_moe=bool(args.use_moe),
        inference_rope_scaling = args.inference_rope_scaling
    )

    model = MiniMindForCausalLM(lm_config)

    use_moe = f"_moe" if args.use_moe else ""
    weight_path = f"{args.save_dir}/{args.weight}_{args.hidden_size}{use_moe}.pth"

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.half().eval().to(args.device)

    total_parameters = sum(p.numel() for p in model.parameters()) / 1e6  # 修改：p.numel 是方法，要加括号
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

    # 修改：使用 optimum.gptq 的 GPTQQuantizer
    quantizer = GPTQQuantizer(
        bits=args.bits,
        dataset=datas,
        group_size=args.group_size,
        damp_percent=0.1,
        desc_act=bool(args.desc_act),
        sym=bool(args.sym),
        model_seqlen=512,  # 校准数据最大长度，需与数据实际长度匹配
        block_name_to_quantize="model.layers",  # MiniMind 的 Transformer 层路径；如报错请根据实际模型结构调整
    )

    # 修改：使用 quantize_model，calibration_dataset 直接传文本列表
    quantized_model = quantizer.quantize_model(
        model=model,
        tokenizer=tokenizer
    )

    # 构建对话
    conversation = []
    if not args.use_auto_prompt:
        prompt = input('请输入测试提示词：')
    else:
        prompt = "你是谁"
    conversation.append({"role": "user", "content": prompt})
    # 修改：移除不存在的 open_thinking 参数
    inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
    input_ids = inputs['input_ids']
    generate_output = quantized_model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False
    )
    response = tokenizer.decode(generate_output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"模型生成：{response}")

    # 修改：使用 quantizer.save 保存量化模型（同时保存权重和量化配置）
    save_dir = f"{args.save_dir}/quantized_{args.bits}bit"
    quantizer.save(quantized_model, save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"量化模型已保存在{save_dir}")

def main():
    parser = argparse.ArgumentParser(description="GPTQ量化")

    parser.add_argument('--tokenizer_path', default='model_compat', type=str, help="分词器目录 (默认: model)")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀 (如 full_sft, pretrain)")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构 (0=否, 1=是)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用 RoPE 外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    
    # 量化与测试参数
    parser.add_argument('--group_size', default= 16, type=int, help="分组大小")
    parser.add_argument('--bits', default=4, type=int, help="量化为q几")
    parser.add_argument('--desc_act', default=0, type=int, help="是否使用按列激活顺序")
    parser.add_argument('--use_auto_prompt', default=1, type=int, help="是否使用默认提示词")
    parser.add_argument('--max_new_tokens', default=50, type=int, help="最大生成token数")
    parser.add_argument('--data_path', default=r"D:/minimind-quantization/dataset/lora_medical.jsonl", type=str, help="数据路径")
    parser.add_argument('--sym', default=1, type=int, help="是否使用对称量化")

    args = parser.parse_args()

    model, tokenizer, weight_path = load_model(args)
    datas = dataest_process(args.data_path,tokenizer)
    GPTQ_achieve(args, weight_path, model, tokenizer, datas)

if __name__ == "__main__":
    main()