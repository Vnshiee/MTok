with open("/home/devraj/MTok/external_models/lvr/Gaps_LVR.md", "r", encoding="utf-8") as f:
    glines = f.readlines()

new_glines = []
for l in glines:
    # 1. \textbf{67.0}
    if "extbf{67.0}" in l:
        l = l.replace("    extbf{67.0}", "\\textbf{67.0}").replace("extbf{67.0}", "\\textbf{67.0}")
    # 2. Script paths
    l = l.replace("`finetune_lvr_stage2_3b.sh`", "[`scripts/finetune_lvr_stage2_3b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_3b.sh)")
    l = l.replace("`finetune_lvr_stage2_7b.sh`", "[`scripts/finetune_lvr_stage2_7b.sh`](file:///home/devraj/MTok/external_models/lvr/repo/scripts/finetune_lvr_stage2_7b.sh)")
    # 3. Anchors
    l = l.replace(
        "[`3_lvr.tex` lines 20–24, 52–58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24)",
        "[`3_lvr.tex` lines 20–24](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L20-L24) and [lines 52–58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L52-L58)"
    )
    l = l.replace(
        "[`src/model/qwen_lvr_model.py` lines 953–954, 1045–1049](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954)",
        "[`src/model/qwen_lvr_model.py` lines 953–954](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L953-L954) and [lines 1045–1049](file:///home/devraj/MTok/external_models/lvr/repo/src/model/qwen_lvr_model.py#L1045-L1049)"
    )
    l = l.replace("#L1013-L1055", "#L1013-L1014")
    new_glines.append(l)

with open("/home/devraj/MTok/external_models/lvr/Gaps_LVR.md", "w", encoding="utf-8") as f:
    f.writelines(new_glines)

with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "r", encoding="utf-8") as f:
    tlines = f.readlines()

new_tlines = []
for l in tlines:
    if l.startswith("| **Multimodal Projector**"):
        l = "| **Multimodal Projector** | $proj(\\mathbf{V}) = \\mathbf{V}_T$ | [`3_lvr.tex` L18](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L18) | [`src/train/train_lvr.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/train/train_lvr.py#L43) | Lines 41, 43 | `model.visual.merger`; frozen in Stage 1 & Stage 2 (`freeze_merger=True`) | Match (Frozen) |\n"
    if l.startswith("| **Latent GRPO Policy Loss**"):
        l = "| **Latent GRPO Policy Loss** | $\\mathcal{J}_{\\mathrm{GRPO}}(\\theta)$ | [`3_lvr.tex` L85](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L85) | [`src/trainer/grpo_trainer.py`](file:///home/devraj/MTok/external_models/lvr/repo/src/trainer/grpo_trainer.py#L740-L757) | Lines 740–757, 1058–1085 | Masked clipped surrogate policy gradient objective | Architectural Discrepancy |\n"
    l = l.replace("lines 309–347, 855–1070", "lines 855–1095")
    l = l.replace("lines 953–1050", "lines 953–1095")
    new_tlines.append(l)

with open("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md", "w", encoding="utf-8") as f:
    f.writelines(new_tlines)

print("Applied exact string substitutions without regex template errors.")
