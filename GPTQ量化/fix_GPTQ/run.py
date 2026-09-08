import os
import json
import torch

from transformers import AutoTokenizer
from optimum.gptq import load_quantized_model

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCausalLM,
)


# ============================================================
# 路径
# ============================================================

ROOT = r"D:\minimind-quantization"

# GPTQ 修复后的目录
GPTQ_DIR = os.path.join(
    ROOT,
    "out",
    "quantized_4bit_fixed"
)

# 原始 FP16
FP16_PATH = os.path.join(
    ROOT,
    "out",
    "full_sft_768.pth"
)


# ============================================================
# 加载原始 FP16 embedding
# ============================================================

def load_fp16_embedding():

    print("=" * 50)
    print("加载原始 FP16 embedding...")
    print("=" * 50)

    state_dict = torch.load(
        FP16_PATH,
        map_location="cpu"
    )

    key = "model.embed_tokens.weight"

    if key not in state_dict:
        raise RuntimeError(
            f"full_sft_768.pth 中找不到 {key}"
        )

    embedding = (
        state_dict[key]
        .detach()
        .cpu()
        .contiguous()
    )

    print(
        "FP16 embedding shape:",
        tuple(embedding.shape)
    )

    print(
        "FP16 embedding std:",
        embedding.float().std().item()
    )

    return embedding


# ============================================================
# 加载 GPTQ
# ============================================================

def load_gptq_model():

    print("=" * 50)
    print("创建 GPTQ 模型...")
    print("=" * 50)

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

    # 避免旧版本 Optimum / MiniMindConfig
    # 处理 quantization_config 时产生兼容问题
    config_dict.pop(
        "quantization_config",
        None
    )

    config = MiniMindConfig(
        **config_dict
    )

    model = MiniMindForCausalLM(
        config
    )

    print("模型创建成功")

    print("=" * 50)
    print("加载 GPTQ 量化权重...")
    print("=" * 50)

    model = load_quantized_model(
        model,
        save_folder=GPTQ_DIR,
        device_map={"": 0},
        disable_exllama=True,
    )

    model.eval()

    print("GPTQ 权重加载完成")

    # --------------------------------------------------------
    # 检查当前 embedding
    # --------------------------------------------------------

    embedding_before = (
        model.model.embed_tokens.weight
    )

    print()
    print("加载后的 embedding:")

    print(
        "shape:",
        tuple(embedding_before.shape)
    )

    print(
        "min:",
        embedding_before.float().min().item()
    )

    print(
        "max:",
        embedding_before.float().max().item()
    )

    print(
        "mean:",
        embedding_before.float().mean().item()
    )

    print(
        "std:",
        embedding_before.float().std().item()
    )

    # --------------------------------------------------------
    # 恢复真正 FP16 embedding
    # --------------------------------------------------------

    fp_embedding = load_fp16_embedding()

    if (
        tuple(embedding_before.shape)
        !=
        tuple(fp_embedding.shape)
    ):
        raise RuntimeError(
            "GPTQ embedding 与 FP16 embedding shape 不一致"
        )

    print()
    print("=" * 50)
    print("强制恢复 FP16 embedding...")
    print("=" * 50)

    with torch.no_grad():

        model.model.embed_tokens.weight.data.copy_(
            fp_embedding.to(
                device=model.model.embed_tokens.weight.device,
                dtype=model.model.embed_tokens.weight.dtype
            )
        )

    # --------------------------------------------------------
    # 重新建立 tied weight
    # --------------------------------------------------------

    model.lm_head.weight = (
        model.model.embed_tokens.weight
    )

    # --------------------------------------------------------
    # 验证
    # --------------------------------------------------------

    embedding_after = (
        model.model.embed_tokens.weight
    )

    lm_after = (
        model.lm_head.weight
    )

    print()
    print("恢复后的 embedding:")

    print(
        "min:",
        embedding_after.float().min().item()
    )

    print(
        "max:",
        embedding_after.float().max().item()
    )

    print(
        "mean:",
        embedding_after.float().mean().item()
    )

    print(
        "std:",
        embedding_after.float().std().item()
    )

    print()
    print(
        "embedding / lm_head 是否共享:",
        (
            model.model.embed_tokens.weight.data_ptr()
            ==
            model.lm_head.weight.data_ptr()
        )
    )

    # --------------------------------------------------------
    # 与 FP16 比较
    # --------------------------------------------------------

    fp_gpu = fp_embedding.cuda().float()

    actual = (
        embedding_after
        .detach()
        .float()
    )

    diff = actual - fp_gpu

    print()
    print("FP16 embedding vs 当前 GPTQ embedding:")

    print(
        "max abs diff:",
        diff.abs().max().item()
    )

    print(
        "mean abs diff:",
        diff.abs().mean().item()
    )

    print(
        "RMSE:",
        torch.sqrt(
            torch.mean(diff * diff)
        ).item()
    )

    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1),
        fp_gpu.reshape(1, -1),
        dim=1
    ).item()

    print(
        "cosine:",
        cosine
    )

    return model


# ============================================================
# tokenizer
# ============================================================

def load_tokenizer():

    print("=" * 50)
    print("正在加载 tokenizer")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(
        GPTQ_DIR
    )

    print("加载 tokenizer 成功")

    return tokenizer


# ============================================================
# 测试输入
# ============================================================

def build_input(tokenizer):

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "你是谁"
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    print()
    print("=" * 50)
    print("Chat Template")
    print("=" * 50)

    print(prompt)

    encoded = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = (
        encoded.input_ids
        .cuda()
    )

    attention_mask = (
        encoded.attention_mask
        .cuda()
    )

    print()
    print("=" * 50)
    print("Input")
    print("=" * 50)

    print(
        "input_ids shape:",
        input_ids.shape
    )

    print(
        "input_ids:",
        input_ids[0].tolist()
    )

    return (
        prompt,
        input_ids,
        attention_mask
    )


# ============================================================
# 关键：no-cache logits 检查
# ============================================================

@torch.no_grad()
def check_logits(
    model,
    input_ids,
    attention_mask
):

    print()
    print("=" * 50)
    print("No-cache logits 检查")
    print("=" * 50)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False
    )

    logits = (
        output
        .logits[:, -1, :]
    )

    print(
        "logits min:",
        logits.min().item()
    )

    print(
        "logits max:",
        logits.max().item()
    )

    print(
        "logits mean:",
        logits.mean().item()
    )

    print(
        "logits std:",
        logits.std().item()
    )

    values, indices = torch.topk(
        logits,
        20,
        dim=-1
    )

    print()
    print("Top-20:")

    for i in range(20):

        print(
            f"{i:2d} "
            f"id={indices[0, i].item():5d} "
            f"logit={values[0, i].item():10.5f}"
        )

    return logits


# ============================================================
# 生成
# ============================================================

@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    input_ids,
    attention_mask
):

    print()
    print("=" * 50)
    print("开始生成")
    print("=" * 50)

    # --------------------------------------------------------
    # 诊断阶段：
    # 禁止 sampling
    # 禁止 KV cache
    # --------------------------------------------------------

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,

        max_new_tokens=50,

        # ★ 核心
        use_cache=False,

        # ★ 确定性
        do_sample=False,

        temperature=1.0,
        top_p=1.0,
        top_k=0,

        repetition_penalty=1.0,

        eos_token_id=2,
    )

    print()
    print("=" * 50)
    print("Generated IDs")
    print("=" * 50)

    print(
        output_ids[0].tolist()
    )

    response = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    print()
    print("=" * 50)
    print("Response")
    print("=" * 50)

    print(response)

    return output_ids


# ============================================================
# main
# ============================================================

def main():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print(
        "torch:",
        torch.__version__
    )

    # --------------------------------------------------------
    # load model
    # --------------------------------------------------------

    model = load_gptq_model()

    # --------------------------------------------------------
    # 额外打印 QuantLinear
    # --------------------------------------------------------

    q_proj = (
        model.model.layers[0]
        .self_attn.q_proj
    )

    print()
    print(
        "q_proj:",
        type(q_proj)
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

    # --------------------------------------------------------
    # tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # input
    # --------------------------------------------------------

    (
        prompt,
        input_ids,
        attention_mask
    ) = build_input(
        tokenizer
    )

    # --------------------------------------------------------
    # no-cache logits
    # --------------------------------------------------------

    logits = check_logits(
        model,
        input_ids,
        attention_mask
    )

    # --------------------------------------------------------
    # generate
    # --------------------------------------------------------

    generate_answer(
        model,
        tokenizer,
        input_ids,
        attention_mask
    )


if __name__ == "__main__":
    main()