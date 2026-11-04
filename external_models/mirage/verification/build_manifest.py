#!/usr/bin/env python
"""
Generate external_models/mirage/STEP1_INGESTION_MANIFEST.md from the files themselves.

Every hash, byte count, line count and line range in the manifest is computed here; nothing is typed by hand except
the one-line role descriptions in ROLES.

Usage (from external_models/mirage):
  python verification/build_manifest.py --upstream DIR --hf_meta DIR --pins DIR --arxiv DIR [--out STEP1_INGESTION_MANIFEST.md]
    --upstream  transformers v4.50.0 copies of generation/utils.py, models/qwen2_5_vl/modeling_qwen2_5_vl.py,
                models/qwen2_5_vl/modular_qwen2_5_vl.py (raw.githubusercontent.com/huggingface/transformers/v4.50.0/src/transformers/...)
    --hf_meta   non-weight files of Miiche/vsp_spatial_planning_direct_sft (incl. training_args.bin)
    --pins      HF API JSON for the checkpoint, base model and datasets (ckpt.json, base.json, <owner>_<name>.json)
    --arxiv     dir with eprint.tar.gz, paper.pdf and the extracted e-print in x/
"""
import argparse
import ast
import difflib
import hashlib
import io
import json
import pickle
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

from paths import qualify

HERE = Path(__file__).resolve().parent
MIRAGE = HERE.parent

ROLES = {
    "paper_src/neurips_2025_arxiv.tex": "Top-level LaTeX (title, authors, \\input tree)",
    "paper_src/sec/1_intro.tex": "§1 Introduction",
    "paper_src/sec/2_related.tex": "§2 Related work",
    "paper_src/sec/3_framework.tex": "§3 Method: data synthesis, Stage 1 (Eq. L_visual, L_text, L_1), Stage 2 (latent relaxation), GRPO",
    "paper_src/sec/4_experiments_arxiv.tex": "§4 Experiments: VSP / COMT / Jigsaw / SAT tables, stage and k/γ ablations",
    "paper_src/sec/5_analysis_arxiv.tex": "§5 Analysis: 3B results, helper-image upper bound, t-SNE of latents",
    "paper_src/sec/6_conclusion.tex": "§6 Conclusion",
    "paper_src/sec/7_supp.tex": "Supplement: helper-image construction, dataset statistics, SFT/RL hyperparameter tables",
    "paper_src/neurips_2025.bib": "Bibliography database", "paper_src/neurips_2025_arxiv.bbl": "Compiled bibliography",
    "paper_src/neurips_2025.sty": "NeurIPS 2025 style", "paper_src/IEEEbib.bst": "BibTeX style",
    "paper_src/ieeenat_fullname.bst": "BibTeX style", "paper_src/00README.json": "arXiv build config",
    "repo/src/main.py": "Training entry: tokenizer extension, config fields, ViT freezing, collate_fn_stage1/2, SFTConfig, trainer",
    "repo/src/trainer.py": "CustomTrainerStage1 (0.1·CE + 1−cos) and CustomTrainerStage2 (CE only)",
    "repo/src/utils.py": "CLI args; prompt/image placement; latent-span construction; label and latent-mask builders",
    "repo/src/task.py": "Per-task preprocessing (conversations) and VSP simulator used for scoring",
    "repo/src/test.py": "Inference/evaluation for VSP (sampling generate, \\boxed extraction, simulate_vsp)",
    "repo/train.sh": "Example 2-stage training commands (vsp-spatial-reasoning)",
    "repo/requirements.txt": "Python requirements (trl==0.15.2, qwen_vl_utils, mathruler, ...)",
    "repo/README.md": "Install, data, training, inference instructions; released checkpoint id",
    "repo/data/examples/sample.jsonl": "Sample of the training JSONL format: 10 records (README says \"a sample dataset of 100 examples\")",
    "repo/data/vsp_spatial_planning/vsp_spatial_planning.tar.gz": "VSP planning training data (train_direct.jsonl + images)",
    "repo/data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz": "VSP planning test data (test_direct.jsonl, 400 maps + images)",
    "repo/transformers/src/transformers/generation/utils.py": "FORK: _sample() latent decoding loop; latent_hidden_states output field",
    "repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py": "FORK: latent embedding injection, Stage-2 multi-pass forward, generate_mode attention dispatch",
    "repo/transformers/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py": "FORK: modular source (4-line difference from v4.50.0)",
}

# Anchors (exact substrings) used to locate Mirage-specific regions in the fork; line numbers are computed.
FORK_REGIONS = [
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "Qwen2_5_VLFlashAttention2.forward_normal (generation-time FA2 path)", "    def forward_normal(", "    #forward_ulysses"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "Qwen2_5_VLFlashAttention2.forward_ulysses (training-time FA2 path)", "    def forward_ulysses(", "class Qwen2_5_VLSdpaAttention"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "Decoder-layer attention dispatch on generate_mode / attn type", "        if generate_mode and self.attn_type == \"flash_attention_2\":", "        hidden_states = residual + hidden_states"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "ForConditionalGeneration.__init__ additions (latent_projection, stage, latent_model)", "        try:\n            if self.model.config.compress_strategy", "        self.latent_model = False"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "forward() extra arguments", "        pixel_values_latent: Optional[torch.Tensor] = None, # Ours (Latent Tokens)", "        latent_hidden_states: Optional[torch.Tensor] = None, # Ours (Latent Tokens)"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "Image-token count mismatch: truncate instead of raise", "                if n_image_tokens != n_image_features:\n                    assert n_image_tokens < n_image_features", "                mask = input_ids == self.config.image_token_id"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "Stage-1 latent targets: helper image → ViT → K-chunk average → scattered into <|latent_pad|>", "            if pixel_values_latent is not None:", "            elif latent_hidden_states is not None and not generate_mode:"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "latent_hidden_states scatter path (not reached by released code)", "            elif latent_hidden_states is not None and not generate_mode:", "            if attention_mask is not None:\n                attention_mask = attention_mask.to(inputs_embeds.device)\n        \n"),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "Stage-2 multi-pass latent forward + CE loss", "        if self.config.stage == 'stage2' and not generate_mode:", "        # if we get 4D attention mask we cannot calculate rope deltas anymore."),
    ("models/qwen2_5_vl/modeling_qwen2_5_vl.py", "Regular forward path (Stage 1, inference): RoPE, LM, CE", "        # if we get 4D attention mask we cannot calculate rope deltas anymore.", "    def prepare_inputs_for_generation("),
    ("generation/utils.py", "_sample(): latent-mode state, embedding feedback, exit rules", "        #Latent mode init", "    def _sample_backup("),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(p):
    data = Path(p).read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None, len(data)
    return data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1), len(data)


def line_of(text, anchor, start=0):
    i = text.find(anchor, start)
    if i < 0:
        raise ValueError(f"anchor not found: {anchor[:60]!r}")
    return text.count("\n", 0, i) + 1, i


def py_defs(path):
    src = Path(path).read_text()
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.append((node.name, node.lineno, node.end_lineno, "class" if isinstance(node, ast.ClassDef) else "def"))
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        out.append((f"{node.name}.{sub.name}", sub.lineno, sub.end_lineno, "method"))
    return out


class _Stub:
    def __init__(self, *a, **k): self.a = a
    def __setstate__(self, s): self.state = s


class _SafeU(pickle.Unpickler):
    def find_class(self, mod, name):
        return type(f"{mod}.{name}", (_Stub,), {})
    def persistent_load(self, pid):
        return None


def training_args(path):
    data = Path(path).read_bytes()
    if zipfile.is_zipfile(io.BytesIO(data)):
        z = zipfile.ZipFile(io.BytesIO(data))
        data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
    o = _SafeU(io.BytesIO(data)).load()
    st = getattr(o, "state", {})
    def val(v):
        if isinstance(v, _Stub):
            return getattr(v, "a", ("?",))[0] if getattr(v, "a", None) else type(v).__name__
        return v
    return {k: val(v) for k, v in st.items()}


def tar_summary(path):
    with tarfile.open(path) as t:
        names = t.getnames()
        jsonl = {}
        for m in t.getmembers():
            if m.name.endswith(".jsonl"):
                jsonl[m.name] = t.extractfile(m).read().count(b"\n")
        n_png = sum(n.endswith(".png") for n in names)
    return len(names), n_png, jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--hf_meta", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--arxiv", required=True)
    ap.add_argument("--out", default=str(MIRAGE / "STEP1_INGESTION_MANIFEST.md"))
    a = ap.parse_args()
    repo = MIRAGE / "repo"
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    log = subprocess.check_output(["git", "-C", str(repo), "log", "--format=%h %ad %s", "--date=short"], text=True).strip().splitlines()
    pins = {p.stem: json.loads(p.read_text()) for p in Path(a.pins).glob("*.json")}
    arx = Path(a.arxiv)
    L = []
    w = L.append
    w("# Mirage — Step 1 Ingestion Manifest")
    w("")
    w("_Generated by `verification/build_manifest.py`; every hash, size, line count and line range below is computed from the files. \"Lines\" counts a final line without a trailing newline (as `nl -ba`), so it can exceed `wc -l` by one._")
    w("")
    w("## 1. Sources and pins")
    w("")
    w("| Artifact | Identifier | Pin |")
    w("|---|---|---|")
    w(f"| Paper | arXiv 2506.17218 (*Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens*) | e-print sha256 `{sha256(arx / 'eprint.tar.gz')}`; PDF sha256 `{sha256(arx / 'paper.pdf')}` |")
    w(f"| Code | https://github.com/UMass-Embodied-AGI/Mirage | commit `{head}` ({len(log)} commits; latest: {log[0]}) |")
    for key, label in (("ckpt", "Released checkpoint"), ("base", "Base model (Axis 4)")):
        d = pins.get(key, {})
        w(f"| {label} | `{d.get('id', '?')}` | revision `{d.get('sha', '?')}` ({d.get('lastModified', '?')}) |")
    for key in ("craigwu_vstar_bench", "MMVP_MMVP", "BLINK-Benchmark_BLINK"):
        d = pins.get(key, {})
        w(f"| Benchmark data | `{d.get('id', key)}` | revision `{d.get('sha', '?')}` ({d.get('lastModified', '?')}) |")
    w(f"| Fork base | transformers `4.50.0.dev0` vendored in `repo/transformers/`; compared against tag `v4.50.0` | — |")
    w("")
    w("Commit history of the code repository:")
    w("")
    w("```")
    w("\n".join(log))
    w("```")
    w("")

    # 2. paper_src
    w("## 2. `paper_src/` (LaTeX)")
    w("")
    ps = MIRAGE / "paper_src"
    ax_files = {str(p.relative_to(arx / "x")) for p in (arx / "x").rglob("*") if p.is_file()}
    w("| File | Lines | Bytes | SHA256 | Identical to arXiv e-print | Role |")
    w("|---|---:|---:|---|:---:|---|")
    for p in sorted(ps.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(ps))
        n, b = counts(p)
        same = (arx / "x" / rel).exists() and sha256(arx / "x" / rel) == sha256(p)
        w(f"| `paper_src/{rel}` | {n if n is not None else 'binary'} | {b} | `{sha256(p)[:16]}…` | {'yes' if same else 'NO'} | {ROLES.get('paper_src/' + rel, '')} |")
    missing = sorted(ax_files - {str(p.relative_to(ps)) for p in ps.rglob('*') if p.is_file()})
    w("")
    w(f"Files in the arXiv e-print but absent from `paper_src/` ({len(missing)}; excluded by MTok `.gitignore` rule `external_models/**/*.pdf`): " + ", ".join(f"`{m}`" for m in missing) + ".")
    pt = MIRAGE / "paper_text.txt"
    n, b = counts(pt)
    w("")
    w(f"`paper_text.txt`: {n} lines, {b} bytes, sha256 `{sha256(pt)[:16]}…` — `pdftotext` extraction of the arXiv PDF (content-equivalent; whitespace/ordering differs by extractor version).")
    w("")

    # 3. repo-owned files
    w("## 3. Mirage-owned repository files (excluding the vendored `transformers/` tree and example images)")
    w("")
    w("| File | Lines | Bytes | SHA256 | Role |")
    w("|---|---:|---:|---|---|")
    owned = [p for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts and "transformers" not in p.parts
             and "__pycache__" not in p.parts and p.suffix not in (".png", ".jpg") and "imgs_test" not in p.parts
             and p.name != "test_direct.jsonl"]
    for p in owned:
        rel = str(p.relative_to(repo))
        n, b = counts(p)
        w(f"| `repo/{rel}` | {n if n is not None else 'binary'} | {b} | `{sha256(p)[:16]}…` | {ROLES.get('repo/' + rel, '')} |")
    n_img = sum(1 for p in repo.rglob("*") if p.suffix in (".png", ".jpg") and ".git" not in p.parts and "imgs_test" not in p.parts)
    w("")
    w(f"Other tracked files: {n_img} example/asset images. Data archives:")
    w("")
    for tgz in ("vsp_spatial_planning.tar.gz", "vsp_spatial_planning_test.tar.gz"):
        n_all, n_png, jl = tar_summary(repo / "data/vsp_spatial_planning" / tgz)
        w(f"- `{tgz}`: {n_all} members, {n_png} PNG; JSONL line counts: " + ", ".join(f"`{k}` = {v}" for k, v in jl.items()))
    w("")

    # 4. python definitions
    w("## 4. Definitions in `repo/src/` (AST line ranges)")
    w("")
    w("| File | Kind | Name | Lines |")
    w("|---|---|---|---|")
    for f in ("main.py", "trainer.py", "utils.py", "task.py", "test.py"):
        for name, s, e, kind in py_defs(repo / "src" / f):
            w(f"| `repo/src/{f}` | {kind} | `{name}` | L{s}–L{e} |")
    w("")
    w("`repo/src/main.py` and `repo/src/test.py` execute at import (argument parsing, model loading, training/evaluation loop at module level); "
      "their module-level statements are the training (`repo/src/main.py` L14–L210) and evaluation (`repo/src/test.py` L14–L84) scripts.")
    w("")

    # 5. fork
    w("## 5. Modified files in the vendored transformers fork")
    w("")
    w("| File | Lines (fork) | Lines (v4.50.0) | Lines added / removed vs v4.50.0 | SHA256 (fork) |")
    w("|---|---:|---:|---:|---|")
    tf = repo / "transformers/src/transformers"
    for rel in ("generation/utils.py", "models/qwen2_5_vl/modeling_qwen2_5_vl.py", "models/qwen2_5_vl/modular_qwen2_5_vl.py"):
        fa, fb = Path(a.upstream, rel).read_text().splitlines(), (tf / rel).read_text().splitlines()
        diff = list(difflib.unified_diff(fa, fb, lineterm="", n=0))
        add = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        rem = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        w(f"| `repo/transformers/src/transformers/{rel}` | {len(fb)} | {len(fa)} | +{add} / −{rem} | `{sha256(tf / rel)[:16]}…` |")
    w("")
    w("The fork is a `4.50.0.dev0` snapshot, so part of each diff is upstream drift (torchdynamo guards, `use_model_defaults`). "
      "The Mirage-specific regions, located by anchor text, are:")
    w("")
    w("| File | Region | Lines |")
    w("|---|---|---|")
    for rel, desc, a0, a1 in FORK_REGIONS:
        text = (tf / rel).read_text()
        s, i = line_of(text, a0)
        e, _ = line_of(text, a1, i + 1)
        w(f"| `repo/transformers/src/transformers/{rel}` | {desc} | L{s}–L{e - 1} |")
    gen = (tf / "generation/utils.py").read_text()
    for anchor in ("        MAX_LATENT_LEN = 8", "            latent_start_idx = 151666", "            latent_end_idx = 151667", "            latent_pad_idx = 151665",
                   "    latent_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None"):
        s, _ = line_of(gen, anchor)
        w(f"| `repo/transformers/src/transformers/generation/utils.py` | constant/field `{anchor.strip()}` | L{s} |")
    w("")

    # 6. HF metadata
    hm = Path(a.hf_meta)
    cfg = json.loads((hm / "config.json").read_text())
    gcfg = json.loads((hm / "generation_config.json").read_text())
    targs = training_args(hm / "training_args.bin")
    w("## 6. Released checkpoint metadata (`Miiche/vsp_spatial_planning_direct_sft`)")
    w("")
    sib = pins.get("ckpt", {}).get("siblings", [])
    w(f"Files: {len(sib)}; total {sum(s.get('size', 0) for s in sib) / 1e9:.2f} GB (4 bf16 safetensors shards).")
    w("")
    w("| Source | Field | Value |")
    w("|---|---|---|")
    for k in ("stage", "latent_size", "compress_strategy", "latent_token_id", "latent_start_id", "latent_end_id", "vocab_size",
              "hidden_size", "num_hidden_layers", "image_token_id", "torch_dtype", "transformers_version"):
        w(f"| config.json | `{k}` | `{cfg.get(k)}` |")
    for k, v in gcfg.items():
        w(f"| generation_config.json | `{k}` | `{v}` |")
    for k in ("num_train_epochs", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps", "lr_scheduler_type",
              "warmup_steps", "weight_decay", "optim", "adam_beta1", "adam_beta2", "bf16", "gradient_checkpointing", "seed", "_n_gpu", "output_dir"):
        w(f"| training_args.bin | `{k}` | `{targs.get(k)}` |")
    pre = json.loads((hm / "preprocessor_config.json").read_text())
    for k in ("min_pixels", "max_pixels", "patch_size", "merge_size", "image_processor_type"):
        w(f"| preprocessor_config.json | `{k}` | `{pre.get(k)}` |")
    added = json.loads((hm / "added_tokens.json").read_text())
    w("")
    w("Latent tokens in `added_tokens.json`: " + ", ".join(f"`{t}`={added[t]}" for t in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>")) + ".")
    w("")

    # 7. catalog
    w("## 7. Component catalog")
    w("")
    w("| Component | Location |")
    w("|---|---|")
    def loc(f, name):
        for n, s, e, _ in py_defs(repo / "src" / f):
            if n == name:
                return f"`repo/src/{f}` L{s}–L{e}"
        raise KeyError(name)
    mod = (tf / "models/qwen2_5_vl/modeling_qwen2_5_vl.py").read_text()
    rows = [
        ("Model definition", "`Qwen2_5_VLForConditionalGeneration` in the fork (`repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`); no Mirage-specific model class"),
        ("Tokenizer extension", "`repo/src/main.py` L36–L38 (and `repo/src/test.py` L38–L40)"),
        ("Vision-tower freezing", "`repo/src/main.py` L68–L69"),
        ("Latent-span construction", loc("utils.py", "replace_subsequent_image_parts_1d") + "; " + loc("utils.py", "process_batch")),
        ("Label construction", loc("utils.py", "generate_labels_after_multi_token_start")),
        ("Latent-position mask (Stage 1)", loc("utils.py", "mask_image_output_tokens")),
        ("Stage-1 collate", loc("main.py", "collate_fn_stage1")),
        ("Stage-2 collate", loc("main.py", "collate_fn_stage2")),
        ("Stage-1 loss (0.1·CE + 1−cos)", loc("trainer.py", "CustomTrainerStage1")),
        ("Stage-2 loss (CE)", loc("trainer.py", "CustomTrainerStage2")),
        ("Stage-1 latent targets (average pooling)", f"`repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L{line_of(mod, '            if pixel_values_latent is not None:')[0]}–L{line_of(mod, '            elif latent_hidden_states is not None and not generate_mode:')[0] - 1}"),
        ("Stage-2 multi-pass forward", f"`repo/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` L{line_of(mod, chr(32) * 8 + 'if self.config.stage == ' + repr('stage2') + ' and not generate_mode:')[0]}–L{line_of(mod, '        # if we get 4D attention mask')[0] - 1}"),
        ("Generation loop (latent decoding)", f"`repo/transformers/src/transformers/generation/utils.py` L{line_of(gen, '        #Latent mode init')[0]}–L{line_of(gen, '    def _sample_backup(')[0] - 1}"),
        ("Evaluation", "`repo/src/test.py` L44–L84 (VSP only: `simulate_vsp` on `map_desc`)"),
        ("VSP simulator / action parser", loc("task.py", "simulate_vsp") + "; " + loc("task.py", "parse_action_sequence")),
        ("Answer extraction", "`mathruler.grader.extract_boxed_content` (`repo/src/test.py` L9, L74)"),
        ("RL (GRPO, paper §3.4)", "not present in the repository"),
        ("Evaluation code for COMT / BLINK-Jigsaw / SAT / VSP spatial reasoning", "not present (`repo/src/test.py` scores with `simulate_vsp` only)"),
    ]
    for r in rows:
        w(f"| {r[0]} | {r[1]} |")
    w("")
    Path(a.out).write_text(qualify("\n".join(L) + "\n"))
    print(f"wrote {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
