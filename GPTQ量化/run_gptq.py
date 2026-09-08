import os
import sys
import json
import torch
from transformers import AutoTokenizer
from optimum.gptq import load_quantized_model

sys.path.insert(0,os.path.join(os.path.dirname(__file__), ".."))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


MODEL_DIR = r"D:\minimind-quantization\out"
GPTQ_DIR = r"D:\minimind-quantization\out\quantized_4bit_fixed"
FP16_PATH = os.path.join(
    MODEL_DIR,
    "full_sft_768.pth"
)

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("torch:", torch.__version__)

def build_model():
    config_path = os.path.join(
        GPTQ_DIR,
        "config.json"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:
        config_dict = json.load(f)

    # GPTQ 量化配置单独由 quantize_config.json 提供
    # 这里是因为optimum会自动读取quantize_config.json，防止从config中读取到dict的quantize_config
    config_dict.pop("quantization_config", None)

    config = MiniMindConfig(**config_dict)

    # 不使用 init_empty_weights
    model = MiniMindForCausalLM(config)

    return model

def load_tokenizer():

    print("="*50)
    print("正在加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        GPTQ_DIR
    )
    print("加载tokenizer成功")

    return tokenizer

def load_gqtq_model():

    model = build_model()

    print("模型创建成功")
    print("="*50)
    print("加载gptq量化权重...")

    model = load_quantized_model(
        model,
        save_folder=GPTQ_DIR,
        device_map={"": 0},
        disable_exllama=True
    )

    state_dict = torch.load(
        FP16_PATH,
        map_location="cpu"
    )

    fp_embedding = state_dict[
        "model.embed_tokens.weight"
    ]

    # 禁用梯度计算
    with torch.no_grad():
        # 将现在model里面的embedding_weight复制为fp16的原始embedding_weight
        model.model.embed_tokens.weight.data.copy_(
            fp_embedding.to(
                device=model.model.embed_tokens.weight.device,# 转移到该设备
                dtype=model.model.embed_tokens.weight.dtype # 转为该权重的类型
            )
        )

    model.lm_head.weight = model.model.embed_tokens.weight

    model.eval()

    print(
        "q_proj:",
        type(model.model.layers[0].self_attn.q_proj)
    )

    print(
        "is_quantized:",
        getattr(model, "is_quantized", None)
    )

    print(
    "lm_head:",
    type(model.lm_head)
    )

    print(
        "embed_tokens:",
        type(model.model.embed_tokens)
    )

    print("权重加载成功")
    print("="*50)

    return model

@torch.inference_mode()
def model_generate(model, tokenizer, prompt):

    conversation = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # 与训练/量化阶段保持完全一致
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    print("=" * 50)
    print("Chat Template:")
    print(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    input_ids = inputs["input_ids"].cuda()

    attention_mask = inputs.get("attention_mask")

    if attention_mask is not None:
        attention_mask = attention_mask.cuda()

    print("=" * 50)
    print("Input:")
    print("input_ids shape:", input_ids.shape)
    print("input_ids:", input_ids[0].tolist())

    output_ids = model.generate(
        input_ids=input_ids,
        use_cache=False,
        attention_mask=None,
        max_new_tokens=50,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = input_ids.shape[1]

    generated_ids = output_ids[0, input_length:]

    print("=" * 50)
    print("Generated IDs:")
    print(generated_ids.tolist())

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("=" * 50)
    print("Response:")
    print(response)

    return response

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuda 不可用"
        )

    model = load_gqtq_model()
    tokenizer = load_tokenizer()

    prompt = "你是谁"

    try:
        model_generate(model, tokenizer, prompt)
    except Exception as e:

        print("\n生成失败：")
        print(type(e).__name__)
        print(e)

        raise


if __name__ =="__main__":
    main()