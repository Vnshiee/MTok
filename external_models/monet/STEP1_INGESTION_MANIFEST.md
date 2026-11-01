# Monet (`arXiv:2511.21395`) — Step 1 Forensic Ingestion Manifest

**Audit Target**: Paper 2 — *Monet: Reasoning in Latent Visual Space Beyond Images and Language* (`arXiv:2511.21395`, CVPR 2026 Paper ID 7696)
**Workspace Directory**: `/home/devraj/MTok/external_models/monet/`
**Manifest Scope**: Step 1 Only (Complete Paper PDF, Official LaTeX Source, GitHub Codebase & HuggingFace Checkpoint Metadata Ingestion)

---

## Section 1: Paper & LaTeX Source Inventory

### 1.1 Bibliographic & Archive Metadata
- **arXiv Identifier**: `arXiv:2511.21395` (`v1` submitted 2025-11-26, e-print timestamp `2025-11-27 12:47:00 UTC`).
- **Title**: *Monet: Reasoning in Latent Visual Space Beyond Images and Language* ([`paper_src/arxiv_main.tex` L49](file:///home/devraj/MTok/external_models/monet/paper_src/arxiv_main.tex#L49)).
- **Conference Acceptance**: CVPR 2026 (Paper ID `7696`, [`arxiv_main.tex` L44-L46](file:///home/devraj/MTok/external_models/monet/paper_src/arxiv_main.tex#L44-L46)).
- **Authors & Affiliations** ([`arxiv_main.tex` L64-L77](file:///home/devraj/MTok/external_models/monet/paper_src/arxiv_main.tex#L64-L77)):
  1. **Qixun Wang** (Peking University — `novaglow646@gmail.com`)
  2. **Yang Shi** (Peking University & Kling Team)
  3. **Yifei Wang** (Amazon AGI SF Lab, work done at MIT)
  4. **Yuanxing Zhang** (Kling Team)
  5. **Pengfei Wan** (Kling Team)
  6. **Kun Gai** (Kling Team)
  7. **Xianghua Ying** (Peking University — Corresponding Author)
  8. **Yisen Wang** (Peking University — Corresponding Author)
- **Top-Level Ingested Paper Artifacts**:

| Artifact Path | `wc -l` (Lines) | Size (Bytes) | SHA256 Checksum | Details |
| :--- | ---: | ---: | :--- | :--- |
| [`Monet_2511.21395.pdf`](file:///home/devraj/MTok/external_models/monet/Monet_2511.21395.pdf) | N/A (Binary) | 22,106,465 | `fba3460fe9d0d97f8bd7f85d778b7053f907666488d23b6b373e5bfc6e8b27bc` | Official arXiv PDF, **18 pages** |
| [`paper_src.tar.gz`](file:///home/devraj/MTok/external_models/monet/paper_src.tar.gz) | N/A (Binary) | 21,280,014 | `c5e37f949b9fb5b5bff87e10352e1b3fc135f9c27320869045a79218593782fc` | Official arXiv e-print source tarball (31 extracted files) |
| [`paper_text.txt`](file:///home/devraj/MTok/external_models/monet/paper_text.txt) | 2,058 | 68,818 | `29cda67e5d466641befe5419dd50399de8225951c10533bc3d570817cd7640fe` | Full plain-text extraction of PDF via `pdftotext` |

### 1.2 Complete File Inventory of `paper_src/` (All 31 Files)

| File Path | Lines | Chars | Size (Bytes) | SHA256 Checksum |
| :--- | ---: | ---: | ---: | :--- |
| [`paper_src/00README.json`](file:///home/devraj/MTok/external_models/monet/paper_src/00README.json) | 17 | 310 | 310 | `b687b554b92ca0dcab250a484c2ddaa5bd6d599358a82505539131a6ca034ce5` |
| [`paper_src/arxiv.sty`](file:///home/devraj/MTok/external_models/monet/paper_src/arxiv.sty) | 508 | 17,969 | 17,969 | `2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153` |
| [`paper_src/arxiv_main.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/arxiv_main.tex) | 104 | 3,210 | 3,210 | `09e03a4f7537a5f8bd3b97f31f8ea2e84450f7d6f05ab79f7df6227082296f63` |
| [`paper_src/figures/SFT2.pdf`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/SFT2.pdf) | N/A (Binary) | N/A (Binary) | 1,690,154 | `7f87b75d0264f0b032142c1e950e58c131d6d70850b8a1015e46d86de47b66d4` |
| [`paper_src/figures/dataset_construction.pdf`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/dataset_construction.pdf) | N/A (Binary) | N/A (Binary) | 4,918,447 | `c82b93b3afdf37a700cc49927b045897f996601e1e850a86915ea0aac0aa9dc5` |
| [`paper_src/figures/latent_size.pdf`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/latent_size.pdf) | N/A (Binary) | N/A (Binary) | 39,037 | `f850aabdb2225954fe5e253abb43c1a08aba2090de6253c9c8c30664c17201c7` |
| [`paper_src/figures/obs_acc.pdf`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/obs_acc.pdf) | N/A (Binary) | N/A (Binary) | 17,405 | `a40f23ee1785a66cd08a2d1b82c29793f9e1f1a86ef7e54405d9b90bf1da0a22` |
| [`paper_src/figures/overview.pdf`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/overview.pdf) | N/A (Binary) | N/A (Binary) | 1,258,899 | `3140dbaa8a8a0c50f9b5d62a2ca5b759d2ca9995fd82098aed0118843afa8a65` |
| [`paper_src/figures/training_data_examples/1.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/1.png) | N/A (Binary) | N/A (Binary) | 2,171,746 | `49ff47b8bd45fe757bf6cfb09d31aae494ddd6c332619fcd74b4fe7a2cb2434c` |
| [`paper_src/figures/training_data_examples/10.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/10.png) | N/A (Binary) | N/A (Binary) | 2,333,482 | `260347700f096f171ab2ca7470f01d1d4f2f79c78cda3c09a27f1daac757d765` |
| [`paper_src/figures/training_data_examples/11.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/11.png) | N/A (Binary) | N/A (Binary) | 5,085 | `93a5780391aa94766149edaf6098f98ed37cbef8b929bc890e52f9cbd393d18e` |
| [`paper_src/figures/training_data_examples/19.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/19.png) | N/A (Binary) | N/A (Binary) | 112,528 | `ac2aa0fc1ddc5d8f38f748f3c69505dbd8db6705418a5ae9602ed856a6f3bc00` |
| [`paper_src/figures/training_data_examples/2.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/2.png) | N/A (Binary) | N/A (Binary) | 2,164,089 | `e9898a21ce5d44e0b91d9f2a75955d7734793c6df3b99d1a75545f8396a66fd2` |
| [`paper_src/figures/training_data_examples/20.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/20.png) | N/A (Binary) | N/A (Binary) | 220,069 | `39781b190a84a060aa985b8ac282ee8e0646541a80ea9d9a71c55b6391e6cb76` |
| [`paper_src/figures/training_data_examples/21.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/21.png) | N/A (Binary) | N/A (Binary) | 945,229 | `b270f78f7dc56c0d0fbcff1108bd1d35f13f84f75b4d2d7d802b11efc5658170` |
| [`paper_src/figures/training_data_examples/22.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/22.png) | N/A (Binary) | N/A (Binary) | 427,852 | `de42ff5add380bd2f643af9639e2c1da3b98ff9a0ab3a26a62303f872ee65a62` |
| [`paper_src/figures/training_data_examples/23.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/23.png) | N/A (Binary) | N/A (Binary) | 3,044,564 | `9b664f0250f0b7a26fb4db281e3ea03f326fc1838593f5ba6213264671836161` |
| [`paper_src/figures/training_data_examples/3.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/3.png) | N/A (Binary) | N/A (Binary) | 2,155,243 | `9bac3dc1bddeb5ffcb4f66632605ce41f92e34fdeb6562c6dc8d8a554edb9d64` |
| [`paper_src/figures/training_data_examples/5.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/5.png) | N/A (Binary) | N/A (Binary) | 125,031 | `a5bbc531c3799d9e1725d1ccb35277f6fadaa5fdb9f66aad8f2bac0f49f67078` |
| [`paper_src/figures/training_data_examples/6.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/6.png) | N/A (Binary) | N/A (Binary) | 130,828 | `68cf925d28cf501649a02a7021b7e17d6ae90e98587c25322c3efee50eae8fbd` |
| [`paper_src/figures/training_data_examples/7.png`](file:///home/devraj/MTok/external_models/monet/paper_src/figures/training_data_examples/7.png) | N/A (Binary) | N/A (Binary) | 161,173 | `b9766d3863caaa23fd2ab352d8e30bd72f75b57b0ff692af9c1ac124b1be39d4` |
| [`paper_src/ieeenat_fullname.bst`](file:///home/devraj/MTok/external_models/monet/paper_src/ieeenat_fullname.bst) | 1,448 | 26,900 | 26,900 | `e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc` |
| [`paper_src/main.bib`](file:///home/devraj/MTok/external_models/monet/paper_src/main.bib) | 439 | 18,524 | 18,525 | `0b02a35bf288590e03ebe05c3b66ec591d0ccb97d2f568bdb04404957455a95b` |
| [`paper_src/preamble.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/preamble.tex) | 51 | 1,380 | 1,380 | `f666a85903f49929395951e7afb620e96d4e208334623f23135298b4d7f06228` |
| [`paper_src/sec/0_abstract.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/sec/0_abstract.tex) | 5 | 1,941 | 1,947 | `9da867944cd42a01edcc9e718df181f5bc2d7e1bf5e6710318891dba2156dea6` |
| [`paper_src/sec/1_intro.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/sec/1_intro.tex) | 18 | 5,460 | 5,460 | `200c9eaa55bc1a1322b47bcebff6b5e64c2f995ab15b94f348c7d8312f58b76b` |
| [`paper_src/sec/2_related_work.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/sec/2_related_work.tex) | 19 | 3,372 | 3,374 | `096941b6ec3a1c672eac4170d9b2bd7f5b7f4fd695467525ac3b64177df44531` |
| [`paper_src/sec/3_method.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/sec/3_method.tex) | 201 | 22,679 | 22,705 | `d2652b06f08f6f78dd0a2924c47b6c77ee37cef9fc0fed634de7f2e8b3ede6d7` |
| [`paper_src/sec/4_experiment.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/sec/4_experiment.tex) | 145 | 13,224 | 13,226 | `d585ccf42b125b0d264e041fc658263a9640f38c8f9f116dd8c9ba1052282004` |
| [`paper_src/sec/5_conclusion.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/sec/5_conclusion.tex) | 5 | 1,065 | 1,065 | `df8eecc4f49564824e8af29ba9711df70e300d58330fd38ebe0dc8e7623f0e63` |
| [`paper_src/sec/X_suppl.tex`](file:///home/devraj/MTok/external_models/monet/paper_src/sec/X_suppl.tex) | 473 | 26,041 | 26,091 | `e90e4fd43c6982ea724696395008b5e99b19e6483fd038377f1fa0259230dc06` |

---

## Section 2: Code Repository Inventory (`external_models/monet/repo`)

### 2.1 Git Metadata
- **Repository URL**: `https://github.com/NOVAglow646/Monet.git`
- **Local Clone Path**: `/home/devraj/MTok/external_models/monet/repo`
- **Current Commit Hash**: `08939996996d934ae8e8095cb9dca4566f09e755` (`HEAD -> main`)
- **Commit Message**: `fix inference example`
- **Total Tracked Files**: 127 non-git files across `monet_qwen_model/`, `src/`, `RL/`, `inference/`, `script_examples/`, and `deepspeed/`

### 2.2 Key Module File Table

| File Path | Lines | Size (Bytes) | SHA256 Checksum | Description |
| :--- | ---: | ---: | :--- | :--- |
| [`repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py`](file:///home/devraj/MTok/external_models/monet/repo/monet_qwen_model/modeling_qwen2_5_vl_monet.py) | 2,583 | 129,384 | `6978e3fe893f9474c9634b66064ec8c879b7a94ee5a3ce9c7b8887339def5632` | Core SFT & inference monkey-patched Qwen2.5-VL model with latent embedding injection |
| [`repo/src/trainer.py`](file:///home/devraj/MTok/external_models/monet/repo/src/trainer.py) | 350 | 16,468 | `8877c251488391d8194dd5cb19086eb7f29bacfed55a04912eb47386fadc1376` | SFT Trainer implementing student-teacher alignment loss and detached proxy loss |
| [`repo/src/main.py`](file:///home/devraj/MTok/external_models/monet/repo/src/main.py) | 423 | 20,075 | `5fe8af5115c0b4c4ebc4d99959379fdb912193bd33fccffd9f61216547fcb3cc` | Main training entry point for SFT stages 1, 2, and 3 |
| [`repo/src/precompute_teacher_latents.py`](file:///home/devraj/MTok/external_models/monet/repo/src/precompute_teacher_latents.py) | 331 | 14,355 | `912af5a670ea84f0ccbb13bcd439205643bfe5af0a8b70a2cfa4ce66e03119ad` | Offline precomputation of teacher latent embeddings to .pt tensors |
| [`repo/src/precompute_teacher_reps.py`](file:///home/devraj/MTok/external_models/monet/repo/src/precompute_teacher_reps.py) | 342 | 15,888 | `6b8229e60aad677e992512b889478b9a14b7561b6ceff015c26d8d12e40d3c2c` | Offline precomputation of teacher intermediate representations |
| [`repo/src/task.py`](file:///home/devraj/MTok/external_models/monet/repo/src/task.py) | 96 | 4,690 | `040846f7f40541ced43813171fd948f53b945cd1d5361c62cd6a4cdffe912813` | Data task and prompt collation for Monet SFT |
| [`repo/src/utils.py`](file:///home/devraj/MTok/external_models/monet/repo/src/utils.py) | 919 | 40,907 | `ec29fd66f0a971d5cc961f38b6df8959d0be8738b0aa9907f12c659bc52aa301` | Tensor loading, evaluation, and helper utilities |
| [`repo/RL/monet_rl_patch.py`](file:///home/devraj/MTok/external_models/monet/repo/RL/monet_rl_patch.py) | 80 | 2,582 | `d8beaa2ef631b64ac2a90bc577a4a487c0225b3381020cb5591a8c4c4a3306e4` | RL monkey patch for EasyR1 implementing VLPO policy optimization |
| [`repo/RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py`](file:///home/devraj/MTok/external_models/monet/repo/RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py) | 2,070 | 101,010 | `ec29e25e1174164f1784b14169a79880eb847025ef51fc49922a708f7cf5ea25` | Custom Qwen2.5-VL transformer model adapted for EasyR1 RL |
| [`repo/RL/monet_models/vllm/latent_hook.py`](file:///home/devraj/MTok/external_models/monet/repo/RL/monet_models/vllm/latent_hook.py) | 554 | 18,631 | `5b7e4191578b2be290e839371cb321815653a6b541024906c86b526bd25c0e1f` | vLLM latent hook for generating continuous latent trajectories during RL rollout |
| [`repo/inference/vllm_inference_example.py`](file:///home/devraj/MTok/external_models/monet/repo/inference/vllm_inference_example.py) | 40 | 1,469 | `5b6fe5c0cd8c883fcf806863a0ac3f75ab11e3e115eb2f7b7822b3e48d739653` | Official vLLM generation script using LATENT_SIZE=10 |
| [`repo/inference/vllm/monet_gpu_model_runner.py`](file:///home/devraj/MTok/external_models/monet/repo/inference/vllm/monet_gpu_model_runner.py) | 2,929 | 136,105 | `3b34e478e555ce54a6df80280adca82762bcf09f1da7d08c970eb645561f5a6b` | Custom vLLM gpu_model_runner intercepting token 151666 to inject last-layer states |
| [`repo/script_examples/sft_stage1.sh`](file:///home/devraj/MTok/external_models/monet/repo/script_examples/sft_stage1.sh) | 26 | 1,013 | `4933616ffc9bd05d7d38749b1c9de7de6c951b3f76cce016ee2f8e75c6523741` | Stage 1 standard autoregressive SFT bash script |
| [`repo/script_examples/sft_stage2.sh`](file:///home/devraj/MTok/external_models/monet/repo/script_examples/sft_stage2.sh) | 56 | 2,499 | `cc1b1d240605396c56c7c2d6746633286cd6f934bb26881a95fe4873cc29cdf3` | Stage 2 teacher representation alignment bash script |
| [`repo/script_examples/sft_stage3.sh`](file:///home/devraj/MTok/external_models/monet/repo/script_examples/sft_stage3.sh) | 60 | 2,821 | `1f1b2b1f4a79340bd335145a4c3b6014cd6a0c2ed8277150a29702524d616fa7` | Stage 3 pure latent thought generation bash script |
| [`repo/RL/examples/vlpo_train.sh`](file:///home/devraj/MTok/external_models/monet/repo/RL/examples/vlpo_train.sh) | 90 | 3,188 | `47fe3e769d2b025b036e116d9416e2fe3bf1482c0f5540c2355358ecece61192` | Stage 4 VLPO reinforcement learning script based on EasyR1 |
| [`repo/README.md`](file:///home/devraj/MTok/external_models/monet/repo/README.md) | 248 | 11,014 | `65786204303d9183130bb4ffce04aa29e9621bf077ab979037cdff3b8734ea14` | Official project documentation and usage guide |
| [`repo/requirements.txt`](file:///home/devraj/MTok/external_models/monet/repo/requirements.txt) | 26 | 263 | `12e8e3e370415f7648152debd93837f10708de6e508ce16c4a169260220d8bf5` | Project dependencies |

---

## Section 3: HuggingFace Checkpoint & Dataset Inventory

### 3.1 Primary Model Checkpoint: `NOVAglow646/Monet-7B`
- **HuggingFace Repository ID**: `NOVAglow646/Monet-7B`
- **Commit SHA**: `39c1e320a3e15c357bff911fda263923edbdda37`
- **Total Siblings / Files**: 17
- **Base Backbone**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Model Architecture**: `Qwen2_5_VLForConditionalGeneration` ($L=28$ decoder layers, $D=3584$ hidden dimension, $H_Q=28$ query heads, $H_{KV}=4$ KV heads)
- **Vocabulary Size**: `151,670` tokens (151,643 standard text + 27 multimodal/special tokens)
- **Multimodal & Latent Special Tokens**:
  - Token `151665`: `<abs_vis_token_pad>`
  - Token `151666`: `<abs_vis_token>` (Latent Reasoning Start Token — triggers latent mode switch)
  - Token `151667`: `</abs_vis_token>` (Latent Reasoning End Token)
  - Token `151668`: `<observation>`
  - Token `151669`: `</observation>`

### 3.2 SFT Intermediate Checkpoint: `NOVAglow646/Monet-SFT-7B`
- **HuggingFace Repository ID**: `NOVAglow646/Monet-SFT-7B`
- **Commit SHA**: `67919cada4e1e6bbc013819a4d483d9bb914d629`
- **Total Siblings / Files**: 49
- **Description**: Stage 3 SFT checkpoint prior to VLPO reinforcement learning.

### 3.3 Training Dataset: `NOVAglow646/Monet-SFT-125K`
- **HuggingFace Dataset ID**: `NOVAglow646/Monet-SFT-125K`
- **Total Samples**: ~125,000 multi-turn visual reasoning dialogues constructed from Zebra-CoT, Visual-CoT, CogCoM, and ReFoCus.

---

## Section 4: AST, Tensor & Architectural Comparison Matrix

| Dimension | Monet (`arXiv:2511.21395`) | LVR (`arXiv:2509.24251`) | Mull-Tokens (CVPR 2026) |
| :--- | :--- | :--- | :--- |
| **Base Model** | `Qwen2.5-VL-7B` ($L=28, D=3584$) | `Qwen2.5-VL-7B` ($L=28, D=3584$) | `Qwen2.5-VL-7B` ($L=28, D=3584$) |
| **Latent Trigger Token** | `<abs_vis_token>` (`151666`) | `<|lvr_start|>` (`151665`) | `<|thought_start|>` |
| **Latent Terminator Token** | `</abs_vis_token>` (`151667`) | `<|lvr_end|>` (`151668`) | `<|thought_end|>` |
| **Latent Representation Source** | Last-layer hidden state $\mathbf{h}^{(L)}$ | Last-layer hidden state $\mathbf{h}^{(L)}$ (optional projection head) | Soft prompt embeddings |
| **Inference Latent Size** | Fixed quota `LATENT_SIZE=10` (default) | Quota-based $K \in \{4, 8, 16\}$ | Fixed 20 slots |
| **SFT Supervision Target** | Precomputed offline teacher representations (`.pt` files) | Online ViT ROI patch tokens ($\|h - v\|_2^2$) | Cross-entropy answer supervision |
| **RL Policy Gradient** | VLPO via detached surrogate $\sum (v \cdot g.\text{detach}())$ in `src/trainer.py` L39 | GRPO with zero-gradient masked latents (`final_mask=0`) | GRPO with $G=2$ group cancellation |

---

## Section 5: Step 1 Forensic Verification Checklist

- [x] Paper PDF verified (arXiv:2511.21395, 18 pages, SHA256 verified).
- [x] LaTeX source unpacked and verified (31 files including all sections, tables, and figures).
- [x] GitHub repository cloned and clean (`HEAD` commit `0893999`).
- [x] HuggingFace metadata ingested for `Monet-7B`, `Monet-SFT-7B`, and `Monet-SFT-125K`.
- [x] Special tokens (`151665`–`151669`) verified against tokenizer vocabulary.
- [x] Code architecture analyzed across SFT (`modeling_qwen2_5_vl_monet.py`), RL (`monet_rl_patch.py`), and vLLM inference (`monet_gpu_model_runner.py`).
- [x] Checksum and byte integrity confirmed for all artifacts.
