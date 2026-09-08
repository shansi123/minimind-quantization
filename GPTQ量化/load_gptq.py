import os
import sys
import json
import torch
from optimum.gptq import load_quantized_model

sys.path.insert(0,os.path.join(os.path.dirname(__file__), ".."))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


MODEL_DIR = r"D:\minimind-quantization\out"
GPTQ_DIR = r"D:\minimind-quantization\out\quantized_4bit"

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

def main():

    print("="*50)
    print("正在加载空模型...")

    model = build_model()

    print("模型创建成功")
    print("="*50)
    print("查看量化前 Linear")
    print(
        "q_proj:",
        type(model.model.layers[0].self_attn.q_proj)
    )

    print(
        "k_proj:",
        type(model.model.layers[0].self_attn.k_proj)
    )

    print(
        "v_proj:",
        type(model.model.layers[0].self_attn.v_proj)
    )

    print(
        "o_proj:",
        type(model.model.layers[0].self_attn.o_proj)
    )

    print("="*50)
    print("加载gptq量化权重...")

    model = load_quantized_model(
        model,
        save_folder=GPTQ_DIR,
        device_map={"": 0}
    )

    print("权重加载成功")
    print("="*50)
    print("查看量化后 Linear")
    print(
        "q_proj:",
        type(model.model.layers[0].self_attn.q_proj)
    )

    print(
        "k_proj:",
        type(model.model.layers[0].self_attn.k_proj)
    )

    print(
        "v_proj:",
        type(model.model.layers[0].self_attn.v_proj)
    )

    print(
        "o_proj:",
        type(model.model.layers[0].self_attn.o_proj)
    )

    print()

    print(
        "is_quantized =",
        getattr(model, "is_quantized", None)
    )

if __name__ =="__main__":
    main()