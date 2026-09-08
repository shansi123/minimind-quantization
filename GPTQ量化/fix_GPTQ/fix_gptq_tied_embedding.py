import os
import json

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from optimum.gptq import load_quantized_model


# ============================================================
# 路径
# ============================================================

ROOT = r"D:\minimind-quantization"

FP16_PATH = os.path.join(
    ROOT,
    "out",
    "full_sft_768.pth"
)

GPTQ_DIR = os.path.join(
    ROOT,
    "out",
    "quantized_4bit"
)


# ============================================================
# 工具
# ============================================================

def stat(name, x):

    x = x.detach().float().cpu()

    print()
    print(name)
    print("shape:", tuple(x.shape))
    print("min:", x.min().item())
    print("max:", x.max().item())
    print("mean:", x.mean().item())
    print("std:", x.std().item())


def compare(name, a, b):

    a = a.detach().float().cpu()
    b = b.detach().float().cpu()

    diff = a - b

    mae = diff.abs().mean().item()
    rmse = torch.sqrt(
        torch.mean(diff * diff)
    ).item()

    cosine = torch.nn.functional.cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1),
        dim=1
    ).item()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("max abs diff:",
          diff.abs().max().item())

    print("mean abs diff:",
          mae)

    print("RMSE:",
          rmse)

    print("cosine:",
          cosine)


# ============================================================
# 加载 FP16
# ============================================================

def load_fp16():

    print("=" * 70)
    print("加载 FP16")
    print("=" * 70)

    print(FP16_PATH)

    state_dict = torch.load(
        FP16_PATH,
        map_location="cpu"
    )

    config = MiniMindConfig()

    model = MiniMindForCausalLM(
        config
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(
        "missing:",
        len(missing)
    )

    print(
        "unexpected:",
        len(unexpected)
    )

    if missing:
        print(missing)

    if unexpected:
        print(unexpected)

    model = model.half()
    model.cuda()
    model.eval()

    return model


# ============================================================
# 加载 GPTQ
# ============================================================

def load_gptq():

    print()
    print("=" * 70)
    print("加载 GPTQ")
    print("=" * 70)

    with open(
        os.path.join(
            GPTQ_DIR,
            "config.json"
        ),
        "r",
        encoding="utf-8"
    ) as f:

        config_dict = json.load(f)

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

    model = load_quantized_model(
        model,
        save_folder=GPTQ_DIR,
        device_map={"": 0},
        disable_exllama=True,
    )

    model.eval()

    return model


# ============================================================
# 输入
# ============================================================

def build_input_ids():

    # 与之前诊断完全相同的 prompt
    input_ids = torch.tensor(
        [[
            1,
            832,
            311,
            234,
            441,
            357,
            677,
            259,
            2,
            234,
            1,
            1388,
            570,
            811,
            234,
            25,
            234,
            234,
            26,
            234,
            234
        ]],
        dtype=torch.long,
        device="cuda"
    )

    return input_ids


# ============================================================
# logits
# ============================================================

@torch.no_grad()
def get_logits(
    model,
    input_ids
):

    output = model(
        input_ids=input_ids,
        use_cache=False
    )

    return output.logits[:, -1, :]


# ============================================================
# top-k
# ============================================================

def show_topk(
    name,
    logits,
    k=20
):

    values, indices = torch.topk(
        logits,
        k
    )

    print()
    print(name)

    for i in range(k):

        print(
            f"{i:2d} "
            f"id={indices[0, i].item():5d} "
            f"logit={values[0, i].item():10.5f}"
        )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print("GPTQ tied embedding 修复验证")
    print("=" * 70)

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
    # 模型
    # --------------------------------------------------------

    fp16 = load_fp16()

    gptq = load_gptq()

    # --------------------------------------------------------
    # 原始 embedding
    # --------------------------------------------------------

    fp_embedding = (
        fp16
        .model
        .embed_tokens
        .weight
        .detach()
        .clone()
    )

    fp_lm_head = (
        fp16
        .lm_head
        .weight
        .detach()
        .clone()
    )

    gptq_embedding_before = (
        gptq
        .model
        .embed_tokens
        .weight
        .detach()
        .clone()
    )

    gptq_lm_before = (
        gptq
        .lm_head
        .weight
        .detach()
        .clone()
    )

    # --------------------------------------------------------
    # 修复前
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("修复前")
    print("=" * 70)

    stat(
        "FP16 embedding",
        fp_embedding
    )

    stat(
        "GPTQ embedding",
        gptq_embedding_before
    )

    stat(
        "FP16 lm_head",
        fp_lm_head
    )

    stat(
        "GPTQ lm_head",
        gptq_lm_before
    )

    compare(
        "FP16 embedding vs GPTQ embedding",
        fp_embedding,
        gptq_embedding_before
    )

    # --------------------------------------------------------
    # prompt
    # --------------------------------------------------------

    input_ids = build_input_ids()

    # --------------------------------------------------------
    # FP16 logits
    # --------------------------------------------------------

    logits_fp16 = get_logits(
        fp16,
        input_ids
    )

    # --------------------------------------------------------
    # GPTQ 修复前 logits
    # --------------------------------------------------------

    logits_before = get_logits(
        gptq,
        input_ids
    )

    stat(
        "FP16 logits",
        logits_fp16
    )

    stat(
        "GPTQ logits BEFORE fix",
        logits_before
    )

    compare(
        "FP16 logits vs GPTQ BEFORE fix",
        logits_fp16,
        logits_before
    )

    show_topk(
        "FP16 top-20",
        logits_fp16
    )

    show_topk(
        "GPTQ BEFORE fix top-20",
        logits_before
    )

    # --------------------------------------------------------
    # ★ 核心修复
    #
    # 使用原始 FP16 embedding
    # 恢复 GPTQ embedding
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("执行 tied embedding 修复")
    print("=" * 70)

    with torch.no_grad():

        gptq.model.embed_tokens.weight.data.copy_(
            fp_embedding
        )

    # --------------------------------------------------------
    # 重新 tie
    #
    # 你的 MiniMind 本身就是：
    #
    # model.embed_tokens.weight
    # =
    # lm_head.weight
    #
    # --------------------------------------------------------

    with torch.no_grad():
        gptq.model.embed_tokens.weight.data.copy_(fp_embedding)

    gptq.lm_head.weight = gptq.model.embed_tokens.weight

    # --------------------------------------------------------
    # 修复后检查
    # --------------------------------------------------------

    gptq_embedding_after = (
        gptq
        .model
        .embed_tokens
        .weight
    )

    gptq_lm_after = (
        gptq
        .lm_head
        .weight
    )

    print()
    print("=" * 70)
    print("修复后 embedding / lm_head")
    print("=" * 70)

    stat(
        "GPTQ embedding AFTER fix",
        gptq_embedding_after
    )

    stat(
        "GPTQ lm_head AFTER fix",
        gptq_lm_after
    )

    print()
    print(
        "GPTQ embedding / lm_head 是否共享:",
        (
            gptq.model.embed_tokens.weight.data_ptr()
            ==
            gptq.lm_head.weight.data_ptr()
        )
    )

    compare(
        "FP16 embedding vs GPTQ embedding AFTER fix",
        fp_embedding,
        gptq_embedding_after
    )

    compare(
        "FP16 lm_head vs GPTQ lm_head AFTER fix",
        fp_lm_head,
        gptq_lm_after
    )

    # --------------------------------------------------------
    # 修复后 logits
    # --------------------------------------------------------

    logits_after = get_logits(
        gptq,
        input_ids
    )

    stat(
        "GPTQ logits AFTER fix",
        logits_after
    )

    compare(
        "FP16 logits vs GPTQ AFTER fix",
        logits_fp16,
        logits_after
    )

    show_topk(
        "GPTQ AFTER fix top-20",
        logits_after
    )

    print()
    print("=" * 70)
    print("修复验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()