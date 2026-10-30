with open("/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("| **(1)** |"):
        new_lines.append("| **(1)** | [`sections/3_lvr.tex` L55-L58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L55-L58) | `equation` (unlabeled) | $\\mathcal{L}_{\\mathrm{LVR}} = \\frac{1}{T_v} \\sum_{t=1}^{T_v} \\big\\| \\mathbf{h}_t - \\mathbf{v}_t \\big\\|_2^2$ |\n")
    elif line.startswith("| **(2)** |"):
        new_lines.append("| **(2)** | [`sections/3_lvr.tex` L63-L67](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L63-L67) | `equation` (unlabeled) | $\\mathcal{L}_{\\mathrm{NTP}} = - \\frac{1}{T_y} \\sum_{t=1}^{T_y} \\log p_\\theta(y_t \\mid y_{<t}, \\mathbf{h}_{1:T_v})$ |\n")
    elif line.startswith("| **(3)** |"):
        new_lines.append("| **(3)** | [`sections/3_lvr.tex` L71-L73](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L71-L73) | `equation` (unlabeled) | $\\mathcal{L} = \\mathcal{L}_{\\mathrm{NTP}} + \\lambda_{\\mathrm{LVR}} \\cdot \\mathcal{L}_{\\mathrm{LVR}}$ |\n")
    elif line.startswith("| **(4)** |"):
        new_lines.append("| **(4)** | [`sections/3_lvr.tex` L85-L96](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L85-L96) | `align` (unlabeled) | $J_{\\mathrm{GRPO_{latent}}}(\\theta) = \\mathbb{E}_{q, I, o \\sim \\pi_{\\theta_{\\mathrm{old}}}} \\left[ \\frac{1}{|y|} \\sum_{t=1}^{|y|} \\min \\left( r_{i,t}(\\theta) \\hat{A}_{i,t}, \\operatorname{clip}(r_{i,t}(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_{i,t} \\right) - \\beta D_{\\mathrm{KL}}(\\pi_\\theta \\| \\pi_{\\mathrm{ref}}) \\right]$ |\n")
    elif line.startswith("| **(5)** |"):
        new_lines.append("| **(5)** | [`sections/3_lvr.tex` L98-L100](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L98-L100) | `equation` (unlabeled) | $r_{i,t}(\\theta) = \\frac{\\pi_{\\theta}(y_{i,t}\\mid q, I, \\widetilde{h}^{\\mathrm{latent}}_{i}, y_{i,<t})}{\\pi_{\\theta_{\\mathrm{old}}}(y_{i,t}\\mid q, I, \\widetilde{h}^{\\mathrm{latent}}_{i}, y_{i,<t})}$ |\n")
    elif line.startswith("| **(6)–(7)** |"):
        new_lines.append("| **(6)** | [`sections/3_lvr.tex` L111-L116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116) | `align` (unlabeled) | $\\tilde{R}_i = \\frac{R(y_i)-\\mathrm{mean}(R(y_{1}),\\dots,R(y_{G}))}{\\mathrm{std}(R(y_{1}),\\dots,R(y_{G}))}, \\qquad \\hat{A}_{i,t} = \\tilde{R}_i$ |\n")
    else:
        new_lines.append(line)

with open("/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Direct line replace of equations finished.")
