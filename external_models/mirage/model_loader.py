"""
Model loader and intervention hooks for the Mirage ablation study.

Mirage (arXiv 2506.17218, repo UMass-Embodied-AGI/Mirage @ 53f26de) ships a modified
transformers 4.50.0.dev0 under `repo/transformers/src`. Its Qwen2_5_VLForConditionalGeneration.forward
takes an extra `generate_mode` flag: when False and `config.stage == 'stage2'` the forward runs the
multi-pass Stage-2 training path (modeling_qwen2_5_vl.py L2163-L2358), so every inference call here
passes `generate_mode=True`, exactly as the fork's own `_sample` loop does (generation/utils.py L3317, L3320).

This module:
  * puts the fork (and the repo's `src/` helpers) first on `sys.path` without pip-installing anything;
  * loads the target checkpoint (`Miiche/vsp_spatial_planning_direct_sft`) or the un-finetuned base model;
  * reads and cross-checks the latent token ids against the ids hard-coded in the fork's decoding loop;
  * installs forward hooks that capture (a) the last-layer, post-final-norm hidden state, which is what the
    fork feeds back as the next latent input (`outputs.hidden_states`, modeling L2410, L2438) and
    (b) the merged visual-token embeddings produced by `model.visual` at prefill;
  * provides `LatentIntervention`, which decides the vector fed at each latent position.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

MIRAGE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = MIRAGE_DIR / "repo"

# Constants hard-coded in the fork's decoding loop (repo/transformers/src/transformers/generation/utils.py L3370-L3372)
FORK_LATENT_START_ID = 151666
FORK_LATENT_END_ID = 151667
FORK_LATENT_PAD_ID = 151665
FORK_MAX_LATENT_LEN = 8          # generation/utils.py L3295

TARGET_MODEL_ID = "Miiche/vsp_spatial_planning_direct_sft"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def activate_mirage_fork(repo_dir: Path = DEFAULT_REPO_DIR, require_fork: bool = True):
    """Put the Mirage transformers fork and repo/src on sys.path (must run before `import transformers`)."""
    repo_dir = Path(repo_dir).resolve()
    fork_src = repo_dir / "transformers" / "src"
    repo_src = repo_dir / "src"
    if "transformers" in sys.modules and require_fork:
        loaded = Path(sys.modules["transformers"].__file__).resolve()
        if fork_src not in loaded.parents:
            raise RuntimeError(f"transformers already imported from {loaded}; activate_mirage_fork() must run first")
    for p in (str(repo_src), str(fork_src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import transformers  # noqa: E402

    is_fork = Path(transformers.__file__).resolve().is_relative_to(fork_src) if fork_src.exists() else False
    if require_fork and not is_fork:
        raise RuntimeError(
            f"Mirage fork not found at {fork_src}. Clone it first:\n"
            f"  git clone https://github.com/UMass-Embodied-AGI/Mirage.git {repo_dir}\n"
            f"  git -C {repo_dir} checkout 53f26de2682025e146781c5b198ec93bdfe4c4d6"
        )
    return transformers, is_fork


@dataclass
class LatentIds:
    pad: int
    start: int
    end: int
    eos: List[int]
    newline: int


def resolve_latent_ids(tokenizer, config=None) -> LatentIds:
    """Read latent token ids from the tokenizer and assert they match the config and the fork constants."""
    ids = {name: tokenizer.convert_tokens_to_ids(tok) for name, tok in
           (("pad", "<|latent_pad|>"), ("start", "<|latent_start|>"), ("end", "<|latent_end|>"))}
    unk = tokenizer.unk_token_id
    if any(v is None or v == unk for v in ids.values()):
        raise ValueError(f"latent tokens missing from tokenizer: {ids}")
    if (ids["pad"], ids["start"], ids["end"]) != (FORK_LATENT_PAD_ID, FORK_LATENT_START_ID, FORK_LATENT_END_ID):
        raise ValueError(f"tokenizer latent ids {ids} differ from the ids hard-coded in the fork decoding loop")
    if config is not None:
        for name, attr in (("pad", "latent_token_id"), ("start", "latent_start_id"), ("end", "latent_end_id")):
            if getattr(config, attr, ids[name]) != ids[name]:
                raise ValueError(f"config.{attr}={getattr(config, attr)} != tokenizer id {ids[name]}")
    eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline) == 1, newline
    return LatentIds(pad=ids["pad"], start=ids["start"], end=ids["end"], eos=eos, newline=newline[0])


class ForwardCapture:
    """Forward hook that keeps the most recent output of a module."""

    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.value = output[0] if isinstance(output, tuple) else output

    def remove(self):
        self.handle.remove()


@dataclass
class LoadedModel:
    model: object
    processor: object
    tokenizer: object
    model_mode: str
    model_path: str
    is_fork: bool
    accepts_generate_mode: bool
    latent_ids: Optional[LatentIds]
    final_hidden: ForwardCapture
    visual_out: ForwardCapture
    device: str
    dtype: object
    info: Dict = field(default_factory=dict)

    def forward(self, **kwargs):
        """Model forward; adds generate_mode=True on the fork so Stage-2 training code never runs at inference."""
        if self.accepts_generate_mode:
            kwargs["generate_mode"] = True
        return self.model(**kwargs, return_dict=True)

    def last_hidden(self):
        """[B, T, H] last-layer post-norm hidden states of the most recent forward."""
        return self.final_hidden.value

    def input_embeddings(self):
        return self.model.get_input_embeddings()


def load_model(model_path: str, model_mode: str = "target", device: str = "cuda", dtype: str = "bfloat16",
               attn_implementation: str = "sdpa", min_pixels: Optional[int] = None, max_pixels: Optional[int] = None,
               repo_dir: Path = DEFAULT_REPO_DIR, require_fork: bool = True, cache_dir: Optional[str] = None) -> LoadedModel:
    transformers, is_fork = activate_mirage_fork(repo_dir, require_fork=require_fork)
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    proc_kwargs = {}
    if min_pixels is not None:
        proc_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        proc_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    if proc_kwargs:
        # Qwen2VLImageProcessor resizes with size["shortest_edge"/"longest_edge"] (fork image_processing_qwen2_vl.py
        # L147-L153, L256-L257); kwargs to AutoProcessor.from_pretrained do not reach it, so set them explicitly.
        ip = processor.image_processor
        ip.size = dict(ip.size)
        if min_pixels is not None:
            ip.size["shortest_edge"] = ip.min_pixels = min_pixels
        if max_pixels is not None:
            ip.size["longest_edge"] = ip.max_pixels = max_pixels
    tokenizer = processor.tokenizer
    if model_mode == "target":
        # src/test.py L38-L40 re-adds the latent tokens after loading (a no-op when the checkpoint tokenizer has them)
        for tok in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"):
            tokenizer.add_tokens(tok, special_tokens=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation, cache_dir=cache_dir,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    latent_ids = None
    if model_mode == "target":
        latent_ids = resolve_latent_ids(tokenizer, model.config)
        if getattr(model.config, "latent_size", None) is None:
            raise ValueError("target checkpoint has no config.latent_size; is this a Mirage checkpoint?")
    accepts_generate_mode = "generate_mode" in inspect.signature(model.forward).parameters
    if model_mode == "target" and require_fork and not accepts_generate_mode:
        raise RuntimeError("loaded forward has no generate_mode argument; the Mirage fork is not active")

    final_hidden = ForwardCapture(model.model.norm)
    visual_out = ForwardCapture(model.visual)
    info = {
        "transformers_version": transformers.__version__,
        "transformers_file": transformers.__file__,
        "is_mirage_fork": is_fork,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "config_stage": getattr(model.config, "stage", None),
        "config_latent_size": getattr(model.config, "latent_size", None),
        "config_compress_strategy": getattr(model.config, "compress_strategy", None),
        "vocab_size": model.config.vocab_size,
        "processor_max_pixels": getattr(processor.image_processor, "max_pixels", None),
        "processor_min_pixels": getattr(processor.image_processor, "min_pixels", None),
        "attn_implementation": attn_implementation,
        "dtype": dtype,
    }
    return LoadedModel(model=model, processor=processor, tokenizer=tokenizer, model_mode=model_mode,
                       model_path=model_path, is_fork=is_fork, accepts_generate_mode=accepts_generate_mode,
                       latent_ids=latent_ids, final_hidden=final_hidden, visual_out=visual_out,
                       device=str(model.device), dtype=torch_dtype, info=info)


def stable_seed(text: str, salt: int = 0) -> int:
    return int(hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()[:8], 16)


class LatentIntervention:
    """
    Decides the vector fed at each latent position.

    `natural` is the vector Mirage itself would feed: the last-layer post-norm hidden state of the previous
    position (fork generation/utils.py L3417-L3419 sets `inputs_embeds = outputs.hidden_states`).

    modes:
      none   C0: feed `natural`
      zero   C1: feed a zero vector
      noise  C2: feed isotropic Gaussian noise rescaled to ||natural|| (seeded per question id and step)
      donor  C3: feed the natural latent recorded at the same step for a different question (donor bank)
      pause  extra control: feed the input embedding of a fixed text token (default ".") -- pause-token control
      mean   extra control: feed the per-step mean of the donor bank (dataset-average latent)
    """

    MODES = ("none", "zero", "noise", "donor", "pause", "mean")

    def __init__(self, mode: str, loaded: Optional[LoadedModel] = None, donor_bank: Optional[Dict[str, "object"]] = None,
                 donor_map: Optional[Dict[str, str]] = None, pause_token: str = ".", noise_salt: int = 0):
        if mode not in self.MODES:
            raise ValueError(f"unknown intervention {mode}")
        self.mode = mode
        self.donor_bank = donor_bank or {}
        self.donor_map = donor_map or {}
        self.noise_salt = noise_salt
        self.pause_vec = None
        self.mean_by_step = None
        if mode == "pause":
            import torch
            tid = loaded.tokenizer.encode(pause_token, add_special_tokens=False)
            assert len(tid) == 1, f"pause token {pause_token!r} must be a single token, got {tid}"
            with torch.no_grad():
                self.pause_vec = loaded.input_embeddings()(torch.tensor(tid, device=loaded.model.device))[0]
        if mode == "mean":
            import torch
            if not self.donor_bank:
                raise ValueError("mean intervention needs a donor bank of natural latents")
            max_steps = max(v.shape[0] for v in self.donor_bank.values())
            self.mean_by_step = []
            for s in range(max_steps):
                rows = [v[s] for v in self.donor_bank.values() if v.shape[0] > s]
                self.mean_by_step.append(torch.stack(rows).float().mean(0))
        if mode == "donor" and not self.donor_bank:
            raise ValueError("donor intervention needs a donor bank (run intervention=none first)")

    def donor_for(self, qid: str) -> Optional[str]:
        return self.donor_map.get(qid)

    def apply(self, qid: str, step: int, natural):
        import torch
        if self.mode == "none":
            return natural, {}
        if self.mode == "zero":
            return torch.zeros_like(natural), {}
        if self.mode == "noise":
            g = torch.Generator(device="cpu").manual_seed(stable_seed(qid, self.noise_salt * 1000 + step))
            noise = torch.randn(natural.shape, generator=g, dtype=torch.float32)
            noise = noise / noise.norm() * natural.float().norm().cpu()
            return noise.to(natural.device, natural.dtype), {}
        if self.mode == "pause":
            return self.pause_vec.to(natural.dtype).view_as(natural), {}
        if self.mode == "mean":
            s = min(step, len(self.mean_by_step) - 1)
            return self.mean_by_step[s].to(natural.device, natural.dtype).view_as(natural), {"mean_step_used": s}
        if self.mode == "donor":
            did = self.donor_for(qid)
            bank = self.donor_bank.get(did)
            if bank is None or bank.shape[0] <= step:
                return torch.zeros_like(natural), {"donor_id": did, "donor_missing_step": True}
            return bank[step].to(natural.device, natural.dtype).view_as(natural), {"donor_id": did}
        raise AssertionError


def build_donor_map(items: List[dict], seed: int = 20261103) -> Dict[str, str]:
    """
    Deterministic donor assignment: each question gets a donor from the same benchmark that is a different
    question and, when possible, has a different ground-truth key (so the donor latent encodes a different answer).
    """
    import random
    rng = random.Random(seed)
    ids = [it["question_id"] for it in items]
    gt = {it["question_id"]: str(it.get("gt_key", it.get("ground_truth"))) for it in items}
    order = ids[:]
    rng.shuffle(order)
    mapping = {}
    for i, qid in enumerate(order):
        donor = None
        for j in range(1, len(order)):
            cand = order[(i + j) % len(order)]
            if cand != qid and gt[cand] != gt[qid]:
                donor = cand
                break
        if donor is None:
            donor = order[(i + 1) % len(order)] if len(order) > 1 else qid
        mapping[qid] = donor
    return mapping
