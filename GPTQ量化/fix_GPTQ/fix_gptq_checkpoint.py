import os
import json
import shutil

import torch

from safetensors import safe_open
from safetensors.torch import save_file


# ============================================================
# 路径
# ============================================================

ROOT = r"D:\minimind-quantization"

FP16_PATH = os.path.join(
    ROOT,
    "out",
    "full_sft_768.pth"
)

SRC_GPTQ_DIR = os.path.join(
    ROOT,
    "out",
    "quantized_4bit"
)

DST_GPTQ_DIR = os.path.join(
    ROOT,
    "out",
    "quantized_4bit_fixed"
)

SRC_MODEL = os.path.join(
    SRC_GPTQ_DIR,
    "model.safetensors"
)

DST_MODEL = os.path.join(
    DST_GPTQ_DIR,
    "model.safetensors"
)


# ============================================================
# 工具
# ============================================================

def stat(name, x):

    x = x.detach().float().cpu()

    print()
    print(name)
    print("shape:", tuple(x.shape))
    print("dtype:", x.dtype)
    print("min:", x.min().item())
    print("max:", x.max().item())
    print("mean:", x.mean().item())
    print("std:", x.std().item())


def compare(name, a, b):

    a = a.detach().float().cpu()
    b = b.detach().float().cpu()

    diff = a - b

    cosine = torch.nn.functional.cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1),
        dim=1,
    ).item()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

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

    print(
        "cosine:",
        cosine
    )


# ============================================================
# 获取原 GPTQ safetensors metadata
# ============================================================

def read_original_metadata():

    print("=" * 70)
    print("读取原始 GPTQ safetensors metadata")
    print("=" * 70)

    with safe_open(
        SRC_MODEL,
        framework="pt",
        device="cpu"
    ) as f:

        metadata = f.metadata()

        keys = list(f.keys())

    print(
        "metadata:"
    )

    print(
        metadata
    )

    print(
        "\ntensor 数量:",
        len(keys)
    )

    print(
        "前 20 个 tensor:"
    )

    for key in keys[:20]:
        print(
            " ",
            key
        )

    if metadata is None:

        raise RuntimeError(
            "原 GPTQ model.safetensors 本身没有 metadata。"
        )

    return metadata, keys


# ============================================================
# 读取原始 GPTQ tensors
# ============================================================

def load_original_gptq():

    print()
    print("=" * 70)
    print("读取原始 GPTQ tensors")
    print("=" * 70)

    tensors = {}

    with safe_open(
        SRC_MODEL,
        framework="pt",
        device="cpu"
    ) as f:

        keys = list(f.keys())

        for key in keys:

            tensors[key] = (
                f.get_tensor(key)
                .contiguous()
            )

    print(
        "读取 tensor 数量:",
        len(tensors)
    )

    return tensors


# ============================================================
# 读取 FP16 embedding
# ============================================================

def load_fp16_embedding():

    print()
    print("=" * 70)
    print("读取原始 FP16 embedding")
    print("=" * 70)

    state_dict = torch.load(
        FP16_PATH,
        map_location="cpu"
    )

    key = "model.embed_tokens.weight"

    if key not in state_dict:

        raise RuntimeError(
            f"full_sft_768.pth 中不存在：{key}"
        )

    embedding = (
        state_dict[key]
        .detach()
        .cpu()
        .contiguous()
    )

    stat(
        "FP16 embedding",
        embedding
    )

    return embedding


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print("修复 GPTQ tied embedding")
    print("=" * 70)

    print("FP16:")
    print(FP16_PATH)

    print("\n原 GPTQ:")
    print(SRC_GPTQ_DIR)

    print("\n新 GPTQ:")
    print(DST_GPTQ_DIR)

    # --------------------------------------------------------
    # 检查
    # --------------------------------------------------------

    if not os.path.exists(FP16_PATH):

        raise FileNotFoundError(
            FP16_PATH
        )

    if not os.path.exists(SRC_MODEL):

        raise FileNotFoundError(
            SRC_MODEL
        )

    # --------------------------------------------------------
    # 创建目标目录
    # --------------------------------------------------------

    if os.path.exists(DST_GPTQ_DIR):

        print()
        print(
            "删除旧的 fixed 目录："
        )

        print(
            DST_GPTQ_DIR
        )

        shutil.rmtree(
            DST_GPTQ_DIR
        )

    os.makedirs(
        DST_GPTQ_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 复制除了 model.safetensors 之外的所有文件
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("复制 GPTQ 配置文件")
    print("=" * 70)

    for name in os.listdir(
        SRC_GPTQ_DIR
    ):

        src = os.path.join(
            SRC_GPTQ_DIR,
            name
        )

        dst = os.path.join(
            DST_GPTQ_DIR,
            name
        )

        if os.path.isfile(src):

            if name == "model.safetensors":
                continue

            shutil.copy2(
                src,
                dst
            )

            print(
                "复制:",
                name
            )

    # --------------------------------------------------------
    # metadata
    # --------------------------------------------------------

    original_metadata, original_keys = (
        read_original_metadata()
    )

    # --------------------------------------------------------
    # tensors
    # --------------------------------------------------------

    tensors = load_original_gptq()

    # --------------------------------------------------------
    # FP16 embedding
    # --------------------------------------------------------

    fp_embedding = load_fp16_embedding()

    embedding_key = (
        "model.embed_tokens.weight"
    )

    # --------------------------------------------------------
    # 当前 GPTQ embedding
    # --------------------------------------------------------

    if embedding_key not in tensors:

        raise RuntimeError(
            "GPTQ checkpoint 中找不到："
            f"{embedding_key}"
        )

    old_embedding = tensors[
        embedding_key
    ]

    stat(
        "原 GPTQ embedding",
        old_embedding
    )

    compare(
        "FP16 embedding vs 原 GPTQ embedding",
        fp_embedding,
        old_embedding
    )

    # --------------------------------------------------------
    # shape 检查
    # --------------------------------------------------------

    if (
        fp_embedding.shape
        !=
        old_embedding.shape
    ):

        raise RuntimeError(
            "\nembedding shape 不匹配：\n"
            f"FP16  = {tuple(fp_embedding.shape)}\n"
            f"GPTQ  = {tuple(old_embedding.shape)}"
        )

    # --------------------------------------------------------
    # ★ 只替换 embedding
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("替换 embedding")
    print("=" * 70)

    tensors[
        embedding_key
    ] = fp_embedding

    # --------------------------------------------------------
    # 检查 tensor key 数量
    # --------------------------------------------------------

    new_keys = list(
        tensors.keys()
    )

    if set(new_keys) != set(original_keys):

        raise RuntimeError(
            "替换 embedding 后 tensor key 集合发生变化。"
        )

    print(
        "Tensor key 集合：保持不变"
    )

    print(
        "Tensor 数量:",
        len(new_keys)
    )

    # --------------------------------------------------------
    # 重要：
    #
    # 必须保留原 GPTQ metadata。
    #
    # 之前 save_file 时自己写 metadata，
    # 导致 Optimum/Accelerate 无法识别。
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("保存新的 model.safetensors")
    print("=" * 70)

    print(
        "使用原 GPTQ metadata:"
    )

    print(
        original_metadata
    )

    save_file(
        tensors,
        DST_MODEL,
        metadata=original_metadata
    )

    print(
        "\n保存完成:"
    )

    print(
        DST_MODEL
    )

    # --------------------------------------------------------
    # 重新读取验证
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("验证新的 safetensors")
    print("=" * 70)

    with safe_open(
        DST_MODEL,
        framework="pt",
        device="cpu"
    ) as f:

        new_metadata = f.metadata()

        new_keys = list(
            f.keys()
        )

        new_embedding = (
            f.get_tensor(
                embedding_key
            )
        )

    print(
        "新文件 metadata:"
    )

    print(
        new_metadata
    )

    if new_metadata != original_metadata:

        raise RuntimeError(
            "\nmetadata 没有正确保留！\n"
            f"原始:\n{original_metadata}\n"
            f"新的:\n{new_metadata}"
        )

    print(
        "\nmetadata 完全一致"
    )

    if set(new_keys) != set(original_keys):

        raise RuntimeError(
            "新文件 tensor keys 与原文件不一致。"
        )

    print(
        "tensor keys 完全一致"
    )

    compare(
        "最终 embedding vs FP16",
        fp_embedding,
        new_embedding
    )

    # --------------------------------------------------------
    # 检查 QuantLinear tensor 数量
    # --------------------------------------------------------

    qkeys = [
        k for k in original_keys
        if (
            ".qweight" in k
            or ".qzeros" in k
            or ".scales" in k
            or ".g_idx" in k
        )
    ]

    print(
        "\nQuantLinear tensor 数量:",
        len(qkeys)
    )

    # --------------------------------------------------------
    # 文件大小
    # --------------------------------------------------------

    print()
    print(
        "原 model.safetensors:",
        f"{os.path.getsize(SRC_MODEL) / 1024 / 1024:.2f} MB"
    )

    print(
        "新 model.safetensors:",
        f"{os.path.getsize(DST_MODEL) / 1024 / 1024:.2f} MB"
    )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("修复成功")
    print("=" * 70)

    print(
        "\n请使用："
    )

    print(
        DST_GPTQ_DIR
    )


if __name__ == "__main__":
    main()