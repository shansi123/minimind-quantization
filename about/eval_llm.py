import time
import argparse  # 命令行解析工具
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):  # 传入默认配置和修改后的相关配置
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)  # 自动加载分词器，根据参数，自动识别并拿到分词器
    if 'model' in args.load_from:  # 这里使用pytorch的原生权重
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))  # 因果掩码模型（自回归生成式模型）
        moe_suffix = '_moe' if args.use_moe else ''  # 根据是否启用moe来选择是否拼接
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'  # 拼接，用于选择权重文件
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:  # 这里使用transformers格式权重
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)  # 获取参数量统计，用告知用户模型大小（如果是moe模式，那么还会额外计算实际激活的参数量）（在加载模型后打印一行日志）
    return model.half().eval().to(args.device), tokenizer  # 返回分词器供后续使用
            # half(): 把模型所有权重转为半精度 float16，显存占用减半、推理更快（MiniMind 部署时的常规做法）。
            # eval(): 切换到推理模式，关闭 Dropout 等训练专属操作，确保生成时输出稳定（不配合 model.train() 使用）。
            # to(): 把模型搬到目标设备（cuda 或 cpu，由 --device 决定）。

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")  # 命令行解析工具
    # 命令行参数（告诉程序要找哪一些权重，以及更改一些初始化设置）
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")  # 通过改变提示词来实现使模型输出think标签
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 创建用于文本流式输出的实例，用于流式输出
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')  # 通过选择是否自动测试
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))  # 保证可复现
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})  # 用户输入
        if 'pretrain' in args.weight:  # 判断使用哪个模式
            inputs = tokenizer.bos_token + prompt  # 如果是预训练模型，则使用文本续写模式
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))  # 如果不是，则使用对话模式
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)  # 通过分词器变为token_id

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)  # 采用解码模式将token_id映射为将要输出的文字
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])  # 显示生成速度
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')  # 格式化输出

if __name__ == "__main__":
    main()