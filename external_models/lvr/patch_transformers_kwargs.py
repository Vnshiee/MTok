path = "/home/devraj/MTok/external_models/lvr/repo/src/train/monkey_patch_forward_lvr.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace import
text = text.replace(
    "from transformers.utils import is_torchdynamo_compiling,TransformersKwargs",
    "from transformers.utils import is_torchdynamo_compiling\ntry:\n    from transformers.utils import TransformersKwargs\nexcept ImportError:\n    TransformersKwargs = dict"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Patched monkey_patch_forward_lvr.py for TransformersKwargs compatibility.")
