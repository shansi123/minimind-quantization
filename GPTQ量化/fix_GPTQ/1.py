from safetensors.torch import load_file

path = r"D:\minimind-quantization\out\quantized_4bit\model.safetensors"

state = load_file(path, device="cpu")

print("keys:")
for key in state.keys():
    if "lm_head" in key or "embed_tokens" in key:
        print(key)

for key in [
    "lm_head.weight",
    "model.embed_tokens.weight",
]:
    if key in state:
        w = state[key].float()

        print(
            key,
            "shape=", tuple(w.shape),
            "min=", w.min().item(),
            "max=", w.max().item(),
            "mean=", w.mean().item(),
            "std=", w.std().item(),
        )