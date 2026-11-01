"""
monet_loader.py:
Provides robust shims and patched model loading for Monet (Qwen2.5-VL-7B backbone).
Handles transformers environment differences (masking_utils, modeling_layers, rope_utils).
"""

import types
import sys
import os
import pathlib
import torch
import torch.nn as nn
from functools import partial

# 1. Setup transformers shims
import transformers.modeling_rope_utils as mru
if not hasattr(mru, 'dynamic_rope_update'):
    def dynamic_rope_update(fn):
        return fn
    mru.dynamic_rope_update = dynamic_rope_update

import transformers.utils as tu
if not hasattr(tu, 'TransformersKwargs'):
    class TransformersKwargs(dict):
        pass
    tu.TransformersKwargs = TransformersKwargs

if not hasattr(tu, 'auto_docstring'):
    def auto_docstring(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def dec(fn):
            return fn
        return dec
    tu.auto_docstring = auto_docstring

if not hasattr(tu, 'can_return_tuple'):
    def can_return_tuple(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def dec(fn):
            return fn
        return dec
    tu.can_return_tuple = can_return_tuple

import transformers.models.qwen2_5_vl.configuration_qwen2_5_vl as qcfg
if not hasattr(qcfg, 'Qwen2_5_VLTextConfig'):
    qcfg.Qwen2_5_VLTextConfig = qcfg.Qwen2_5_VLConfig

# Ensure text_config property exists on Qwen2_5_VLConfig
if not hasattr(qcfg.Qwen2_5_VLConfig, 'text_config'):
    qcfg.Qwen2_5_VLConfig.text_config = property(lambda self: self)

# Ensure layer_types property exists on Qwen2_5_VLConfig
if not hasattr(qcfg.Qwen2_5_VLConfig, 'layer_types'):
    def _get_layer_types(self):
        if getattr(self, "use_sliding_window", False):
            max_win = getattr(self, "max_window_layers", self.num_hidden_layers)
            return ["sliding_attention" if i < max_win else "full_attention" for i in range(self.num_hidden_layers)]
        return ["full_attention"] * self.num_hidden_layers
    qcfg.Qwen2_5_VLConfig.layer_types = property(_get_layer_types)

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# masking_utils shim
if 'transformers.masking_utils' not in sys.modules:
    masking_shim = types.ModuleType('transformers.masking_utils')
    def create_causal_mask(config, input_embeds, attention_mask=None, cache_position=None, past_key_values=None, position_ids=None, **kwargs):
        attn_impl = getattr(config, '_attn_implementation', 'flash_attention_2')
        if attn_impl == 'flash_attention_2':
            if attention_mask is not None and 0 in attention_mask:
                return attention_mask
            return None
        elif attn_impl == 'sdpa':
            if attention_mask is None or (attention_mask is not None and 0 not in attention_mask):
                return None
        bsz, seq_len, _ = input_embeds.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        return _prepare_4d_causal_attention_mask(
            attention_mask,
            sequence_length=seq_len,
            target_length=past_len + seq_len,
            dtype=input_embeds.dtype,
            device=input_embeds.device
        )

    def create_sliding_window_causal_mask(config, input_embeds, attention_mask=None, cache_position=None, past_key_values=None, position_ids=None, **kwargs):
        return create_causal_mask(config, input_embeds, attention_mask=attention_mask, cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids, **kwargs)

    masking_shim.create_causal_mask = create_causal_mask
    masking_shim.create_sliding_window_causal_mask = create_sliding_window_causal_mask
    sys.modules['transformers.masking_utils'] = masking_shim

# modeling_layers shim
if 'transformers.modeling_layers' not in sys.modules:
    layers_shim = types.ModuleType('transformers.modeling_layers')
    class GradientCheckpointingLayer(nn.Module):
        gradient_checkpointing = False
        _can_checkpoint_with_cache = False
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)
    layers_shim.GradientCheckpointingLayer = GradientCheckpointingLayer
    sys.modules['transformers.modeling_layers'] = layers_shim

# 2. Patch transformers Qwen2.5-VL with Monet modeling
# Import original module first so Qwen2_5_VLPreTrainedModel can be imported during exec_module
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as orig_qwen_mod

repo_dir = pathlib.Path(__file__).resolve().parent / "repo"
patch_path = repo_dir / "monet_qwen_model" / "modeling_qwen2_5_vl_monet.py"
import importlib.util
spec = importlib.util.spec_from_file_location("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl_monet", patch_path)
patched_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_mod)

# Now safely replace in sys.modules and top-level transformers symbols
sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = patched_mod
patched_mod.Qwen2_5_VLTextModel.get_input_embeddings = lambda self: self.embed_tokens
patched_mod.Qwen2_5_VLTextModel.set_input_embeddings = lambda self, value: setattr(self, 'embed_tokens', value)

# Update top-level transformers symbols
import transformers
transformers.Qwen2_5_VLForConditionalGeneration = patched_mod.Qwen2_5_VLForConditionalGeneration
transformers.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration = patched_mod.Qwen2_5_VLForConditionalGeneration

from transformers import AutoProcessor, AutoTokenizer

def load_monet_model_and_processor(model_id_or_path="NOVAglow646/Monet-7B", device="cuda:0", torch_dtype=torch.bfloat16):
    """
    Loads Monet-7B model with AutoProcessor and AutoTokenizer.
    Properly handles checkpoint key remapping to guarantee 100% weight loading with strict=True.
    """
    import glob
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig
    
    print(f"[Monet Loader] Resolving weights for {model_id_or_path}...")
    local_dir = snapshot_download(repo_id=model_id_or_path, resume_download=True)
    
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    
    # Monet custom tokens check
    expected_tokens = ["<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>", "<observation>", "</observation>"]
    for t in expected_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"  Token '{t}' -> ID {tid}")
        
    print(f"[Monet Loader] Instantiating architecture (Qwen2.5-VL-7B backbone) directly on {device}...", flush=True)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    with torch.device(device):
        model = patched_mod.Qwen2_5_VLForConditionalGeneration(config)
    torch.set_default_dtype(old_default_dtype)
    
    def remap_key(k):
        if k.startswith('visual.'):
            return 'model.' + k
        elif k.startswith('model.') and not (k.startswith('model.visual.') or k.startswith('model.language_model.')):
            return 'model.language_model.' + k[6:]
        return k
        
    shard_paths = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    print(f"[Monet Loader] Loading {len(shard_paths)} safetensors shards directly to {device}...", flush=True)
    
    full_state_dict = {}
    for p in shard_paths:
        shard = load_file(p, device=device)
        for k, v in shard.items():
            full_state_dict[remap_key(k)] = v
            
    print(f"[Monet Loader] Remapped {len(full_state_dict)} tensors. Executing strict load_state_dict...", flush=True)
    incompatible = model.load_state_dict(full_state_dict, strict=True)
    print(f"[Monet Loader] Strict load passed! Missing keys: {len(incompatible.missing_keys)}, Unexpected: {len(incompatible.unexpected_keys)}", flush=True)
    
    del full_state_dict
    torch.cuda.empty_cache()
    model.eval()
    print(f"[Monet Loader] Successfully loaded Monet model ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params) onto {device}!", flush=True)
    return model, processor, tokenizer

if __name__ == "__main__":
    print("[Monet Loader] Module loaded and shims verified successfully.")
