import os
import json
import math

import torch
import torch.nn as nn

from transformers import AutoTokenizer
from safetensors.torch import load_file
from optimum.gptq import load_quantized_model

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# ============================================================
# 固定路径
# ============================================================

ROOT = r"D:\minimind-quantization"

# 真正的原始 FP16 checkpoint
FP16_PATH = os.path.join(
    ROOT,
    "out",
    "full_sft_768.pth"
)

# GPTQ 输出目录
GPTQ_DIR = os.path.join(
    ROOT,
    "out",
    "quantized_4bit"
)


# ============================================================
# 工具函数
# ============================================================

def print_sep(title=""):
    print()
    print("=" * 72)
    if title:
        print(title)
        print("=" * 72)


def stat_tensor(name, x):
    x = x.detach().float().cpu()

    print()
    print(name)
    print("shape:", tuple(x.shape))
    print("dtype:", x.dtype)
    print("min:", x.min().item())
    print("max:", x.max().item())
    print("mean:", x.mean().item())
    print("std:", x.std().item())


def compare_tensor(name, a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()

    if a.shape != b.shape:
        print_sep(name)
        print("SHAPE 不一致")
        print("A:", tuple(a.shape))
        print("B:", tuple(b.shape))
        return None

    diff = a - b

    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rmse = torch.sqrt(
        torch.mean(diff * diff)
    ).item()

    cosine = torch.nn.functional.cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1),
        dim=1
    ).item()

    print_sep(name)

    print("max abs diff:", max_abs)
    print("mean abs diff:", mean_abs)
    print("RMSE:", rmse)
    print("cosine similarity:", cosine)

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "cosine": cosine,
    }


# ============================================================
# 检查 checkpoint 类型
# ============================================================

def inspect_checkpoint_keys(state_dict):

    print_sep("检查 checkpoint")

    keys = list(state_dict.keys())

    print("checkpoint key 数量:", len(keys))

    print("\n前 30 个 key:")

    for i, key in enumerate(keys[:30]):
        print(" ", key)

    qweight_keys = [
        k for k in keys
        if ".qweight" in k
    ]

    qzero_keys = [
        k for k in keys
        if ".qzeros" in k
    ]

    scale_keys = [
        k for k in keys
        if ".scales" in k
    ]

    linear_weight_keys = [
        k for k in keys
        if k.endswith(".weight")
        and ".qweight" not in k
    ]

    print()
    print("qweight 数量:", len(qweight_keys))
    print("qzeros 数量:", len(qzero_keys))
    print("scales 数量:", len(scale_keys))
    print("普通 weight 数量:", len(linear_weight_keys))

    return (
        qweight_keys,
        qzero_keys,
        scale_keys,
        linear_weight_keys
    )


# ============================================================
# 加载真正的 FP16
# ============================================================

def load_fp16_model():

    print_sep("加载 FP16")

    print("FP16 checkpoint:")
    print(FP16_PATH)

    if not os.path.exists(FP16_PATH):
        raise FileNotFoundError(
            f"\n找不到 FP16 checkpoint:\n{FP16_PATH}"
        )

    print(
        "文件大小:",
        f"{os.path.getsize(FP16_PATH) / 1024 / 1024:.2f} MB"
    )

    # --------------------------------------------------------
    # MiniMind 配置
    # --------------------------------------------------------

    config = MiniMindConfig()

    print("\nMiniMindConfig:")
    print("hidden_size =", config.hidden_size)
    print("num_hidden_layers =", config.num_hidden_layers)
    print("num_attention_heads =", config.num_attention_heads)
    print("num_key_value_heads =", config.num_key_value_heads)
    print("vocab_size =", config.vocab_size)
    print("use_moe =", getattr(config, "use_moe", None))
    print("use_cache =", getattr(config, "use_cache", None))

    # --------------------------------------------------------
    # 创建原始模型
    # --------------------------------------------------------

    model = MiniMindForCausalLM(
        config
    )

    # --------------------------------------------------------
    # 读取 full_sft_768.pth
    # --------------------------------------------------------

    state_dict = torch.load(
        FP16_PATH,
        map_location="cpu"
    )

    # 某些 checkpoint 可能套了一层
    if isinstance(state_dict, dict):

        if (
            "state_dict" in state_dict
            and isinstance(
                state_dict["state_dict"],
                dict
            )
        ):
            print(
                "\n检测到外层 state_dict，展开"
            )
            state_dict = state_dict["state_dict"]

        elif (
            "model" in state_dict
            and isinstance(
                state_dict["model"],
                dict
            )
        ):
            print(
                "\n检测到外层 model，展开"
            )
            state_dict = state_dict["model"]

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "full_sft_768.pth 读取后不是 state_dict"
        )

    (
        qweight_keys,
        qzero_keys,
        scale_keys,
        linear_weight_keys
    ) = inspect_checkpoint_keys(
        state_dict
    )

    # --------------------------------------------------------
    # 这里是非常重要的保护
    # --------------------------------------------------------

    if qweight_keys:
        raise RuntimeError(
            "\n错误：full_sft_768.pth 里面出现了 qweight。\n"
            "这说明你指定的并不是原始 FP16 checkpoint。\n"
            f"发现 {len(qweight_keys)} 个 qweight。"
        )

    if qzero_keys:
        raise RuntimeError(
            "\n错误：full_sft_768.pth 里面出现了 qzeros。\n"
            "这不是原始 FP16 checkpoint。"
        )

    if scale_keys:
        raise RuntimeError(
            "\n错误：full_sft_768.pth 里面出现了 scales。\n"
            "这不是原始 FP16 checkpoint。"
        )

    # --------------------------------------------------------
    # 加载
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print_sep("FP16 load_state_dict")

    print("missing keys:", len(missing))

    for key in missing[:30]:
        print(" ", key)

    if len(missing) > 30:
        print(" ...")

    print("\nunexpected keys:", len(unexpected))

    for key in unexpected[:30]:
        print(" ", key)

    if len(unexpected) > 30:
        print(" ...")

    # --------------------------------------------------------
    # 关键检查
    # --------------------------------------------------------

    weight_missing = [
        x for x in missing
        if x.endswith(".weight")
    ]

    if len(weight_missing) > 10:
        raise RuntimeError(
            "\nFP16 checkpoint 加载异常。\n"
            f"普通 weight 缺失数量：{len(weight_missing)}\n"
            "请检查 full_sft_768.pth 是否与当前 MiniMindConfig 对应。"
        )

    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    model = model.half()
    model.cuda()
    model.eval()

    print("\nFP16 加载成功")

    return model


# ============================================================
# 加载 GPTQ
# ============================================================

def load_gptq_model():

    print_sep("加载 GPTQ")

    config_path = os.path.join(
        GPTQ_DIR,
        "config.json"
    )

    safetensors_path = os.path.join(
        GPTQ_DIR,
        "model.safetensors"
    )

    quant_config_path = os.path.join(
        GPTQ_DIR,
        "quantize_config.json"
    )

    print("GPTQ_DIR:")
    print(GPTQ_DIR)

    print("\nconfig:")
    print(config_path)

    print("\nmodel.safetensors:")
    print(safetensors_path)

    print("\nquantize_config:")
    print(quant_config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"找不到：{config_path}"
        )

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(
            f"找不到：{safetensors_path}"
        )

    # --------------------------------------------------------
    # config
    # --------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:
        config_dict = json.load(f)

    # Optimum 1.19.2 + 当前 MiniMindConfig
    # 避免这里再次处理 quantization_config
    config_dict.pop(
        "quantization_config",
        None
    )

    config = MiniMindConfig(
        **config_dict
    )

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model = MiniMindForCausalLM(
        config
    )

    # --------------------------------------------------------
    # 加载 GPTQ
    # --------------------------------------------------------

    model = load_quantized_model(
        model,
        save_folder=GPTQ_DIR,
        device_map={"": 0},
        disable_exllama=True,
    )

    model.eval()

    print("\nGPTQ 加载成功")

    return model


# ============================================================
# 检查所有 QuantLinear
# ============================================================

def inspect_all_quantlinear(
    gptq_model
):

    print_sep("所有 QuantLinear")

    modules = []

    for name, module in gptq_model.named_modules():

        if module.__class__.__name__ == "QuantLinear":

            modules.append(
                (name, module)
            )

            print(
                name,
                type(module)
            )

    print(
        "\nQuantLinear 数量:",
        len(modules)
    )

    return modules


# ============================================================
# lm_head / embedding
# ============================================================

def inspect_lm_head(
    fp16_model,
    gptq_model
):

    print_sep("检查 lm_head / embedding")

    fp_lm = (
        fp16_model
        .lm_head
        .weight
    )

    g_lm = (
        gptq_model
        .lm_head
        .weight
    )

    fp_emb = (
        fp16_model
        .model
        .embed_tokens
        .weight
    )

    g_emb = (
        gptq_model
        .model
        .embed_tokens
        .weight
    )

    stat_tensor(
        "FP16 lm_head",
        fp_lm
    )

    stat_tensor(
        "GPTQ lm_head",
        g_lm
    )

    stat_tensor(
        "FP16 embedding",
        fp_emb
    )

    stat_tensor(
        "GPTQ embedding",
        g_emb
    )

    compare_tensor(
        "FP16 lm_head vs GPTQ lm_head",
        fp_lm,
        g_lm
    )

    compare_tensor(
        "FP16 embedding vs GPTQ embedding",
        fp_emb,
        g_emb
    )

    # --------------------------------------------------------
    # 是否 tied
    # --------------------------------------------------------

    fp_tied = (
        fp16_model
        .model
        .embed_tokens
        .weight
        .data_ptr()
        ==
        fp16_model
        .lm_head
        .weight
        .data_ptr()
    )

    g_tied = (
        gptq_model
        .model
        .embed_tokens
        .weight
        .data_ptr()
        ==
        gptq_model
        .lm_head
        .weight
        .data_ptr()
    )

    print("\n是否共享权重")
    print("FP16:", fp_tied)
    print("GPTQ:", g_tied)

    if g_tied:
        compare_tensor(
            "GPTQ embedding vs GPTQ lm_head",
            gptq_model.model.embed_tokens.weight,
            gptq_model.lm_head.weight
        )


# ============================================================
# 检查 q_proj 参数
# ============================================================

def inspect_qproj(
    gptq_model
):

    print_sep("检查第 0 层 q_proj")

    q = (
        gptq_model
        .model
        .layers[0]
        .self_attn
        .q_proj
    )

    print(
        "class:",
        type(q)
    )

    print(
        "bits:",
        q.bits
    )

    print(
        "group_size:",
        q.group_size
    )

    print(
        "infeatures:",
        q.infeatures
    )

    print(
        "outfeatures:",
        q.outfeatures
    )

    print(
        "\nqweight:",
        tuple(q.qweight.shape),
        q.qweight.dtype,
        q.qweight.device
    )

    print(
        "qzeros:",
        tuple(q.qzeros.shape),
        q.qzeros.dtype,
        q.qzeros.device
    )

    print(
        "scales:",
        tuple(q.scales.shape),
        q.scales.dtype,
        q.scales.device
    )

    if q.bias is not None:

        print(
            "bias:",
            tuple(q.bias.shape),
            q.bias.dtype,
            q.bias.device
        )

        print(
            "bias min:",
            q.bias.min().item()
        )

        print(
            "bias max:",
            q.bias.max().item()
        )

        print(
            "bias mean:",
            q.bias.float().mean().item()
        )

    if hasattr(q, "g_idx"):

        print(
            "g_idx:",
            tuple(q.g_idx.shape),
            q.g_idx.dtype,
            q.g_idx.device
        )

    return q


# ============================================================
# 4bit unpack
# ============================================================

def unpack_int4(
    packed,
    total_values
):

    """
    把 int32 中打包的 4bit 值展开。

    一个 int32:
        32 / 4 = 8 个值
    """

    if packed.dtype != torch.int32:
        packed = packed.to(torch.int32)

    pack = 8
    mask = 0xF

    rows = []

    for r in range(
        packed.shape[0]
    ):

        value = packed[r]

        for i in range(pack):

            shift = i * 4

            x = (
                value >> shift
            ) & mask

            rows.append(
                x
            )

    out = torch.stack(
        rows,
        dim=0
    )

    return out[
        :total_values
    ]


# ============================================================
# 手工 dequant
# ============================================================

def manual_dequant(
    q
):

    bits = int(q.bits)
    group_size = int(q.group_size)

    if bits != 4:
        raise RuntimeError(
            f"当前脚本只检查 4bit，实际 bits={bits}"
        )

    infeatures = int(q.infeatures)
    outfeatures = int(q.outfeatures)

    n_groups = math.ceil(
        infeatures / group_size
    )

    # --------------------------------------------------------
    # qweight
    # --------------------------------------------------------

    packed_qweight = (
        q.qweight
        .detach()
        .cpu()
    )

    qweight_unpacked = unpack_int4(
        packed_qweight,
        infeatures * outfeatures
    )

    qweight_unpacked = (
        qweight_unpacked
        .reshape(
            infeatures,
            outfeatures
        )
    )

    # --------------------------------------------------------
    # qzeros
    #
    # qzeros:
    #   (48, 96)
    #
    # 48 * 96 * 8
    # =
    # 48 * 768
    #
    # --------------------------------------------------------

    packed_qzeros = (
        q.qzeros
        .detach()
        .cpu()
    )

    qzeros_unpacked = unpack_int4(
        packed_qzeros,
        n_groups * outfeatures
    )

    qzeros_unpacked = (
        qzeros_unpacked
        .reshape(
            n_groups,
            outfeatures
        )
    )

    # --------------------------------------------------------
    # scales
    # --------------------------------------------------------

    scales = (
        q.scales
        .detach()
        .cpu()
        .float()
    )

    scales = scales[
        :n_groups,
        :outfeatures
    ]

    # --------------------------------------------------------
    # dequant
    # --------------------------------------------------------

    weight = torch.empty(
        (
            infeatures,
            outfeatures
        ),
        dtype=torch.float32
    )

    for g in range(
        n_groups
    ):

        start = (
            g * group_size
        )

        end = min(
            start + group_size,
            infeatures
        )

        q_w = (
            qweight_unpacked[
                start:end
            ]
            .float()
        )

        q_z = (
            qzeros_unpacked[g]
            .float()
        )

        scale = scales[g]

        weight[
            start:end
        ] = (
            q_w - q_z
        ) * scale

    # --------------------------------------------------------
    # 当前 weight 是：
    #
    # [infeatures, outfeatures]
    #
    # Linear.weight 应该是：
    #
    # [outfeatures, infeatures]
    #
    # 所以转置
    # --------------------------------------------------------

    weight = (
        weight
        .t()
        .contiguous()
    )

    return weight


# ============================================================
# q_proj 权重比较
# ============================================================

def compare_qproj_weight(
    fp16_model,
    gptq_model
):

    print_sep("q_proj 手工反量化")

    fp_q = (
        fp16_model
        .model
        .layers[0]
        .self_attn
        .q_proj
    )

    g_q = (
        gptq_model
        .model
        .layers[0]
        .self_attn
        .q_proj
    )

    fp_weight = (
        fp_q
        .weight
        .detach()
        .float()
        .cpu()
    )

    dequant_weight = manual_dequant(
        g_q
    )

    stat_tensor(
        "FP16 q_proj weight",
        fp_weight
    )

    stat_tensor(
        "GPTQ manual dequant weight",
        dequant_weight
    )

    compare_tensor(
        "FP16 q_proj weight vs GPTQ dequant weight",
        fp_weight,
        dequant_weight
    )


# ============================================================
# q_proj runtime
# ============================================================

@torch.no_grad()
def compare_qproj_runtime(
    fp16_model,
    gptq_model,
    input_ids
):

    print_sep(
        "FP16 vs GPTQ CUDA：真实 q_proj Runtime"
    )

    # --------------------------------------------------------
    # FP16 layer 0 输入
    # --------------------------------------------------------

    embed = (
        fp16_model
        .model
        .embed_tokens(
            input_ids
        )
    )

    x = (
        fp16_model
        .model
        .layers[0]
        .input_layernorm(
            embed
        )
    )

    x = x.half()

    print(
        "activation:",
        tuple(x.shape),
        x.dtype,
        x.device
    )

    # --------------------------------------------------------
    # FP16 q_proj
    # --------------------------------------------------------

    fp_q = (
        fp16_model
        .model
        .layers[0]
        .self_attn
        .q_proj
    )

    # --------------------------------------------------------
    # GPTQ q_proj
    # --------------------------------------------------------

    g_q = (
        gptq_model
        .model
        .layers[0]
        .self_attn
        .q_proj
    )

    print("\nGPTQ CUDA devices:")

    print(
        "qweight:",
        g_q.qweight.device
    )

    print(
        "qzeros:",
        g_q.qzeros.device
    )

    print(
        "scales:",
        g_q.scales.device
    )

    if g_q.bias is not None:
        print(
            "bias:",
            g_q.bias.device
        )

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------

    y_fp = fp_q(x)

    y_gptq = g_q(x)

    stat_tensor(
        "FP16 q_proj output",
        y_fp
    )

    stat_tensor(
        "GPTQ CUDA q_proj output",
        y_gptq
    )

    compare_tensor(
        "FP16 q_proj output vs GPTQ CUDA q_proj output",
        y_fp,
        y_gptq
    )

    print("\n前 20 个输出值")

    print("\nFP16:")
    print(
        y_fp
        .float()
        .flatten()[:20]
        .cpu()
    )

    print("\nGPTQ CUDA:")
    print(
        y_gptq
        .float()
        .flatten()[:20]
        .cpu()
    )


# ============================================================
# full model logits
# ============================================================

@torch.no_grad()
def compare_full_logits(
    fp16_model,
    gptq_model,
    input_ids
):

    print_sep(
        "FP16 vs GPTQ：Full Model Logits"
    )

    out_fp = fp16_model(
        input_ids=input_ids,
        use_cache=False
    )

    out_gptq = gptq_model(
        input_ids=input_ids,
        use_cache=False
    )

    logits_fp = (
        out_fp
        .logits[:, -1, :]
    )

    logits_gptq = (
        out_gptq
        .logits[:, -1, :]
    )

    stat_tensor(
        "FP16 logits",
        logits_fp
    )

    stat_tensor(
        "GPTQ logits",
        logits_gptq
    )

    compare_tensor(
        "FP16 logits vs GPTQ logits",
        logits_fp,
        logits_gptq
    )

    # --------------------------------------------------------
    # top 20
    # --------------------------------------------------------

    k = 20

    fp_values, fp_indices = torch.topk(
        logits_fp,
        k
    )

    g_values, g_indices = torch.topk(
        logits_gptq,
        k
    )

    print("\nFP16 top-20")

    for i in range(k):

        print(
            f"{i:2d}  "
            f"id={fp_indices[0, i].item():5d}  "
            f"logit={fp_values[0, i].item():10.5f}"
        )

    print("\nGPTQ top-20")

    for i in range(k):

        print(
            f"{i:2d}  "
            f"id={g_indices[0, i].item():5d}  "
            f"logit={g_values[0, i].item():10.5f}"
        )


# ============================================================
# Tokenizer
# ============================================================

def build_input_ids():

    print_sep("Tokenizer / Prompt")

    tokenizer = AutoTokenizer.from_pretrained(
        GPTQ_DIR
    )

    conversation = [
        {
            "role": "user",
            "content": "你是谁"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    print("实际 Prompt:")
    print(prompt)

    encoded = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = (
        encoded.input_ids
        .cuda()
    )

    print(
        "\ninput_ids shape:",
        tuple(input_ids.shape)
    )

    print(
        "input_ids:",
        input_ids[0].tolist()
    )

    return input_ids


# ============================================================
# Main
# ============================================================

def main():

    print_sep("环境信息")

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

    print(
        "FP16:",
        FP16_PATH
    )

    print(
        "GPTQ:",
        GPTQ_DIR
    )

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    fp16_model = load_fp16_model()

    gptq_model = load_gptq_model()

    # --------------------------------------------------------
    # QuantLinear
    # --------------------------------------------------------

    inspect_all_quantlinear(
        gptq_model
    )

    # --------------------------------------------------------
    # lm_head
    # --------------------------------------------------------

    inspect_lm_head(
        fp16_model,
        gptq_model
    )

    # --------------------------------------------------------
    # q_proj
    # --------------------------------------------------------

    inspect_qproj(
        gptq_model
    )

    # --------------------------------------------------------
    # q_proj 权重
    # --------------------------------------------------------

    compare_qproj_weight(
        fp16_model,
        gptq_model
    )

    # --------------------------------------------------------
    # tokenizer
    # --------------------------------------------------------

    input_ids = build_input_ids()

    # --------------------------------------------------------
    # q_proj runtime
    # --------------------------------------------------------

    compare_qproj_runtime(
        fp16_model,
        gptq_model,
        input_ids
    )

    # --------------------------------------------------------
    # full logits
    # --------------------------------------------------------

    compare_full_logits(
        fp16_model,
        gptq_model,
        input_ids
    )

    print_sep("诊断完成")


if __name__ == "__main__":
    main()