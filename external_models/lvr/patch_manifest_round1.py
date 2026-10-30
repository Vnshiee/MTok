with open("/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip_eq = False

for line in lines:
    # Issue 8
    line = line.replace("`QwenGRPOTrainer` (L271-L1578)", "`QwenGRPOTrainer` (L271-L1579)")
    line = line.replace("L271-L1578", "L271-L1579")
    
    # Issue 7
    if "checkpoint_steps = [1500, 2000, 2500]" in line:
        line = '| `evaluation/evaluation.py` | 460 | 460 | 18,921 | `b9b4e60bb277e0c27aa90ea87551d48171e697b8b678a7754cd4f2139a88a932` | CLI entry point for benchmarks (V*, MMVP, BLINK). Sets `STEP_LIST = [4,8,16]` (L27), `CHKPT_PATHS = ""` (L31), and `DECODING_STRATEGY = "steps"` in `main()` (L448). |\n'

    # Issue 6
    line = line.replace('"transformers_version": "4.54.0"', '"transformers_version": "4.54.1"')
    line = line.replace('transformers_version: "4.54.0" (L103)', 'transformers_version: "4.54.1" (L110)')
    line = line.replace('`lvr_head_type`: `"mlp"`', '`lvr_head_type`: `"simple"` (L17)')
    line = line.replace('`loss_mode_switch_fct`: `"mse"` (L26)', '*(Note: `loss_mode_switch_fct` is absent from config.json)*')
    line = line.replace('(L24-L35)', '(L13-L20)')

    # Issue 4
    if "Step 1: `loss = 3.2494`" in line:
        line = "- Step 1: `loss = 12.7922`, `grad_norm = 259.6003`, `learning_rate = 1.3333e-07`\n"
    if "Step 2500 (final): `loss = 0.7796`" in line:
        line = "- Step 2500 (final): `loss = 0.2193`, `grad_norm = 1.0250`, `learning_rate = 0.0`\n"
    line = line.replace("1.1322456598520029e+18", "3883092291223552.0 (3.883e+15)")

    # Issue 5
    line = line.replace('`output_dir`: `"/eb_data/bangzheng/output_models/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128"`',
                        '`output_dir`: `"/dockerx/Local/users/bangzheng/Qwen2.5-VL-7B-Instruct/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-3ug6bgwn"`')
    line = line.replace('`warmup_ratio = 0.1` (250 warmup steps)', '`warmup_ratio = 0.03` (75 warmup steps)')
    line = line.replace('`coconut = True`, `lvr_step_SVG = None`', '*(Note: `coconut` and `lvr_step_SVG` attributes are not present in `training_args.bin`)*')

    # Issue 3: binary files
    for bin_target in ["LVR_2509.24251.pdf", "paper_src.tar.gz", "figures/lvr-teaser.pdf", "figures/lvr_main.pdf", "logos/github.png", "logos/huggingface.png", "logos/web.png", "training_args.bin"]:
        if f"`{bin_target}`" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                # parts: ['', 'filename', 'wc', 'nl', 'bytes', 'sha', 'summary', '']
                parts[2] = "N/A (Binary)"
                parts[3] = "N/A (Binary)"
                line = " | ".join(parts) + "\n"

    # Issue 1: equations
    if "| **Eq. (1)** | `\\mathcal{L}_{CE}" in line:
        skip_eq = True
        new_lines.append('| **Eq. (1)** | `\\mathcal{L}_{\\mathrm{LVR}} = \\frac{1}{T_v} \\sum_{t=1}^{T_v} \\| \\mathbf{h}_t - \\mathbf{v}_t \\|_2^2` | `paper_src/sections/3_lvr.tex` L55-L58 | LVR Latent-to-ROI Visual Feature MSE Loss across $T_v$ latent steps |\n')
        new_lines.append('| **Eq. (2)** | `\\mathcal{L}_{\\mathrm{NTP}} = - \\frac{1}{T_y} \\sum_{t=1}^{T_y} \\log p_\\theta(y_t \\mid y_{<t}, \\mathbf{h}_{1:T_v})` | `paper_src/sections/3_lvr.tex` L63-L67 | Next-Token Prediction Cross-Entropy Loss conditioned on latent states |\n')
        new_lines.append('| **Eq. (3)** | `\\mathcal{L} = \\mathcal{L}_{\\mathrm{NTP}} + \\lambda_{\\mathrm{LVR}} \\cdot \\mathcal{L}_{\\mathrm{LVR}}` | `paper_src/sections/3_lvr.tex` L71-L73 | Total SFT Joint Objective |\n')
        new_lines.append('| **Eq. (4)** | `J_{\\mathrm{GRPO_{latent}}}(\\theta) = \\mathbb{E}_{q, I, o \\sim \\pi_{\\theta_{\\mathrm{old}}}} \\left[ \\frac{1}{|y|} \\sum_{t=1}^{|y|} \\dots \\right]` | `paper_src/sections/3_lvr.tex` L85-L96 | GRPO Policy Optimization Objective over answer tokens |\n')
        new_lines.append('| **Eq. (5)** | `r_{i,t}(\\theta) = \\frac{\\pi_{\\theta}(y_{i,t}\\mid q, I, \\widetilde{h}^{\\mathrm{latent}}_{i}, y_{i,<t})}{\\pi_{\\theta_{\\mathrm{old}}}(y_{i,t}\\mid q, I, \\widetilde{h}^{\\mathrm{latent}}_{i}, y_{i,<t})}` | `paper_src/sections/3_lvr.tex` L98-L100 | Importance Sampling Likelihood Ratio conditioned on latent thoughts |\n')
        new_lines.append('| **Eq. (6)** | `\\tilde{R}_i = \\frac{R(y_i)-\\mathrm{mean}(R(y_1),\\dots,R(y_G))}{\\mathrm{std}(R(y_1),\\dots,R(y_G))}, \\quad \\hat{A}_{i,t} = \\tilde{R}_i` | `paper_src/sections/3_lvr.tex` L111-L116 | Standard Group Advantage Normalization across $G$ completions |\n')
        continue
    if skip_eq:
        if "| **Eq. (" in line:
            continue
        else:
            skip_eq = False

    # Issue 2: Algorithmic claim
    if "- **Combined Group + Type-Specific Advantage" in line:
        new_lines.append("- **Group Advantage Normalization & Latent Token Masking**:\n")
        new_lines.append("  - Standard group advantage normalization is computed at lines 880-892: `advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)`. There is NO sub-group partitioning or type-specific advantage in the code.\n")
        new_lines.append("  - Latent token spans are excluded from gradient updates via token completion masking: tokens between `<|lvr_start|>` and `<|lvr_end|>` are masked out with zero loss weight (`mask = 0`) at lines 740-757.\n")
        new_lines.append("  - Full latent state trajectories are re-evaluated during training via `score_with_lvr_replay` (L1324-L1453) and `_replay_lvr_to_collect_states` (L1483-L1578).\n")
        continue
    if "Partition each prompt's $G$ rollouts into `is_lvr`" in line or "type_advantages" in line or "combined_advantages" in line:
        continue

    new_lines.append(line)

with open("/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Patch applied cleanly.")
