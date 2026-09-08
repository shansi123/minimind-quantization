import os
import sys
import json
import torch
from transformers import AutoTokenizer
from optimum.gptq import load_quantized_model

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCausalLM,
)


FP16_PATH = r"D:\minimind-quantization\out\full_sft_768.pth"
GPTQ_DIR = r"D:\minimind-quantization\out\quantized_4bit"
TOKENIZER_DIR = GPTQ_DIR


def build_fp16_model():

    config = MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        inference_rope_scaling=False,
    )

    model = MiniMindForCausalLM(config)

    state_dict = torch.load(
        FP16_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.half()
    model.cuda()
    model.eval()

    return model


def build_gptq_model():

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

    config_dict.pop(
        "quantization_config",
        None
    )

    config = MiniMindConfig(
        **config_dict
    )

    model = MiniMindForCausalLM(config)

    model = load_quantized_model(
        model,
        save_folder=GPTQ_DIR,
        device_map={"": 0},
        disable_exllama=True,
    )

    model.eval()

    return model


def prepare_input(tokenizer):

    conversation = [
        {
            "role": "user",
            "content": "你是谁"
        }
    ]

    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    return input_ids, attention_mask


@torch.inference_mode()
def get_logits(model, input_ids, attention_mask):

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    return outputs.logits[:, -1, :].float()


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR
    )

    print("加载 FP16...")
    fp16_model = build_fp16_model()

    print("加载 GPTQ...")
    gptq_model = build_gptq_model()

    input_ids, attention_mask = prepare_input(
            tokenizer
        )
    gptq_model.lm_head.weight = torch.nn.Parameter(
        gptq_model.lm_head.weight.detach().clone()
    )

    with torch.no_grad():
        gptq_model.lm_head.weight.copy_(
            fp16_model.lm_head.weight
        )

    gptq_logits = get_logits(
        gptq_model,
        input_ids,
        attention_mask
    )

    print("\n" + "=" * 60)
    print("所有 QuantLinear")
    print("=" * 60)

    for name, module in gptq_model.named_modules():

        if "QuantLinear" in type(module).__name__:

            print(
                name,
                type(module)
            )

    print("\n" + "=" * 60)
    print("检查 lm_head / embedding")
    print("=" * 60)

    fp16_lm = fp16_model.lm_head.weight.detach().float()
    gptq_lm = gptq_model.lm_head.weight.detach().float()

    print("FP16 lm_head:")
    print("shape:", fp16_lm.shape)
    print("min:", fp16_lm.min().item())
    print("max:", fp16_lm.max().item())
    print("mean:", fp16_lm.mean().item())
    print("std:", fp16_lm.std().item())

    print("\nGPTQ lm_head:")
    print("shape:", gptq_lm.shape)
    print("min:", gptq_lm.min().item())
    print("max:", gptq_lm.max().item())
    print("mean:", gptq_lm.mean().item())
    print("std:", gptq_lm.std().item())

    lm_diff = (fp16_lm - gptq_lm).abs()

    print("\nlm_head diff:")
    print("max:", lm_diff.max().item())
    print("mean:", lm_diff.mean().item())

    print("\n是否共享权重:")

    print(
        "FP16:",
        fp16_model.model.embed_tokens.weight.data_ptr()
        == fp16_model.lm_head.weight.data_ptr()
    )

    print(
        "GPTQ:",
        gptq_model.model.embed_tokens.weight.data_ptr()
        == gptq_model.lm_head.weight.data_ptr()
    )

    print("\n" + "=" * 60)
    print("检查第一层 q_proj")
    print("=" * 60)

# 两个模型使用相同 input_ids
    with torch.inference_mode():

        fp16_hidden = fp16_model.model.embed_tokens(
            input_ids
        )

        fp16_hidden = fp16_model.model.layers[0].input_layernorm(
            fp16_hidden
        )

        gptq_hidden = gptq_model.model.embed_tokens(
            input_ids
        )

        gptq_hidden = gptq_model.model.layers[0].input_layernorm(
            gptq_hidden
        )

        fp16_q = fp16_model.model.layers[0].self_attn.q_proj(
            fp16_hidden
        )

        gptq_q = gptq_model.model.layers[0].self_attn.q_proj(
            gptq_hidden
        )

    fp16_q = fp16_q.float()
    gptq_q = gptq_q.float()

    print("FP16 q_proj:")
    print("min:", fp16_q.min().item())
    print("max:", fp16_q.max().item())
    print("mean:", fp16_q.mean().item())
    print("std:", fp16_q.std().item())

    print("\nGPTQ q_proj:")
    print("min:", gptq_q.min().item())
    print("max:", gptq_q.max().item())
    print("mean:", gptq_q.mean().item())
    print("std:", gptq_q.std().item())

    q_diff = (fp16_q - gptq_q).abs()

    print("\nq_proj diff:")
    print("max:", q_diff.max().item())
    print("mean:", q_diff.mean().item())
    print(
        "RMSE:",
        torch.sqrt(
            torch.mean(
                (fp16_q - gptq_q) ** 2
            )
        ).item()
    )

    relative_error = (
        (fp16_q - gptq_q).abs()
        /
        (fp16_q.abs() + 1e-6)
    )

    print(
        "relative mean:",
        relative_error.mean().item()
    )

    print("input shape:", input_ids.shape)

    print()
    print("=" * 60)
    print("计算 FP16 logits")
    print("=" * 60)

    fp16_logits = get_logits(
        fp16_model,
        input_ids,
        attention_mask
    )

    print("FP16 logits:")
    print(
        "min =",
        fp16_logits.min().item()
    )
    print(
        "max =",
        fp16_logits.max().item()
    )
    print(
        "mean =",
        fp16_logits.mean().item()
    )

    print()
    print("=" * 60)
    print("计算 GPTQ logits")
    print("=" * 60)

    gptq_logits = get_logits(
        gptq_model,
        input_ids,
        attention_mask
    )

    print("GPTQ logits:")
    print(
        "min =",
        gptq_logits.min().item()
    )
    print(
        "max =",
        gptq_logits.max().item()
    )
    print(
        "mean =",
        gptq_logits.mean().item()
    )

    print()
    print("=" * 60)
    print("Logits 差异")
    print("=" * 60)

    diff = (
        fp16_logits
        - gptq_logits
    ).abs()

    print(
        "max abs diff =",
        diff.max().item()
    )

    print(
        "mean abs diff =",
        diff.mean().item()
    )

    print(
        "RMSE =",
        torch.sqrt(
            torch.mean(
                (fp16_logits - gptq_logits) ** 2
            )
        ).item()
    )

    print()
    print("=" * 60)
    print("Top 20 token")
    print("=" * 60)

    fp16_top = torch.topk(
        fp16_logits,
        20,
        dim=-1
    )

    gptq_top = torch.topk(
        gptq_logits,
        20,
        dim=-1
    )

    print("FP16:")
    print(fp16_top.indices[0].tolist())

    print("GPTQ:")
    print(gptq_top.indices[0].tolist())

    print()
    print("GPTQ token对应文字:")

    for token_id in gptq_top.indices[0].tolist():
        print(
            token_id,
            repr(
                tokenizer.decode(
                    [token_id]
                )
            )
        )


if __name__ == "__main__":
    main()