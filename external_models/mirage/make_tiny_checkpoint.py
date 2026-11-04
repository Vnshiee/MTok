"""
Build a tiny randomly-initialised Mirage-shaped checkpoint for CPU testing.

It keeps everything that the ablation code depends on from the released checkpoint
(Miiche/vsp_spatial_planning_direct_sft): vocab size 151668, latent token ids 151665/151666/151667,
config.stage='stage2', config.latent_size=4, compress_strategy='average', the real tokenizer, chat template,
image-processor config and generation_config. Only the network is shrunk (2 text layers, hidden 64, 2-block ViT).

Usage: python make_tiny_checkpoint.py <released_ckpt_metadata_dir> <out_dir>
  <released_ckpt_metadata_dir> must contain config.json, generation_config.json and the tokenizer/processor files
  (fetch with: huggingface-cli download Miiche/vsp_spatial_planning_direct_sft --exclude "*.safetensors" ...).
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import model_loader as ML  # noqa: E402

ML.activate_mirage_fork()
import torch  # noqa: E402
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

PROCESSOR_FILES = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "added_tokens.json",
                   "special_tokens_map.json", "preprocessor_config.json", "chat_template.json", "generation_config.json"]


def build(src: Path, out: Path, seed: int = 0):
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, max_window_layers=2, rope_scaling={"mrope_section": [2, 3, 3], "rope_type": "default", "type": "default"})
    cfg["vision_config"].update(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64,
                                fullatt_block_indexes=[1], torch_dtype="float32")
    cfg["torch_dtype"] = "float32"
    config = Qwen2_5_VLConfig.from_dict(cfg)
    torch.manual_seed(seed)
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for f in PROCESSOR_FILES:
        shutil.copy(src / f, out / f)
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1]), Path(sys.argv[2])))
