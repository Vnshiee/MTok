import re

path = "/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Fix Issue 1: Equations table in Section 1.3 B
old_eq_section = """| **(1)** | [`sections/3_lvr.tex` L55-L58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L55-L58) | `equation` (unlabeled) | $\\mathcal{L}_{\\text{CE}} = - \\sum_{t \\in \\mathcal{T}_{\\text{text}}} \\log p_\\theta\\big(y_t \\mid y_{<t}, V, Z_{<t}\\big)$ |
| **(2)** | [`sections/3_lvr.tex` L63-L67](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L63-L67) | `equation` (unlabeled) | $\\mathcal{L}_{\\text{LVR}} = \\frac{1}{\\|\\mathcal{T}_{\\text{lvr}}\\|} \\sum_{t \\in \\mathcal{T}_{\\text{lvr}}} \\big\\| z_t - v^{\\text{roi}}_t \\big\\|_2^2$ |
| **(3)** | [`sections/3_lvr.tex` L71-L73](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L71-L73) | `equation` (unlabeled) | $\\mathcal{L}_{\\text{SFT}} = \\mathcal{L}_{\\text{CE}} + \\lambda_{\\text{lvr}} \\, \\mathcal{L}_{\\text{LVR}}$ |
| **(4)** | [`sections/3_lvr.tex` L85-L96](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L85-L96) | `align` (`\\nonumber` on L86, numbered on L95) | $\\mathcal{J}_{\\mathrm{GRPO}}(\\theta) = \\mathbb{E}_{q \\sim \\mathcal{D},\\, \\{o_i\\}_{i=1}^G \\sim \\pi_{\\theta_{\\text{old}}}(\\cdot \\mid q)} \\Bigg[ \\frac{1}{G} \\sum_{i=1}^G \\frac{1}{\\|o_i\\|} \\sum_{t \\notin [\\text{lvr\\_start}, \\text{lvr\\_end}]} \\Big\\{ \\min\\!\\Big( r_{i,t}(\\theta)\\,\\hat{A}_{i,t}, \\; \\operatorname{clip}\\big(r_{i,t}(\\theta), 1-\\epsilon, 1+\\epsilon\\big)\\,\\hat{A}_{i,t} \\Big) - \\beta \\, D_{\\mathrm{KL}}\\!\\big(\\pi_\\theta \\,\\|\\, \\pi_{\\text{ref}}\\big) \\Big\\} \\Bigg]$ |
| **(5)** | [`sections/3_lvr.tex` L98-L100](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L98-L100) | `equation` (unlabeled) | $\\hat{A}_{i,t} = \\frac{r_i - \\mathrm{mean}(\\mathbf{r})}{\\mathrm{std}(\\mathbf{r})}$ |
| **(6)–(7)** | [`sections/3_lvr.tex` L111-L116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116) | `align` (2 numbered lines; `\\label{eq:adv_text_only}` on L115) | $\\hat{A}_{i,t}^{\\text{type}} = \\frac{r_i - \\mathrm{mean}(\\mathbf{r}_{\\text{type}})}{\\mathrm{std}(\\mathbf{r}_{\\text{type}})}, \\quad \\text{type} \\in \\{\\text{text}, \\text{lvr}\\} (Eq. 6); \\hat{A}_{i,t}^{\\text{balanced}} = \\lambda_{\\text{type}} \\cdot \\hat{A}_{i,t} + (1 - \\lambda_{\\text{type}}) \\cdot \\hat{A}_{i,t}^{\\text{type}}, \\quad \\lambda_{\\text{type}} = \\frac{N_{\\text{type}}}{G} (Eq. 7)$ |"""

new_eq_section = """| **(1)** | [`sections/3_lvr.tex` L55-L58](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L55-L58) | `equation` (unlabeled) | $\\mathcal{L}_{\\mathrm{LVR}} = \\frac{1}{T_v} \\sum_{t=1}^{T_v} \\big\\| \\mathbf{h}_t - \\mathbf{v}_t \\big\\|_2^2$ |
| **(2)** | [`sections/3_lvr.tex` L63-L67](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L63-L67) | `equation` (unlabeled) | $\\mathcal{L}_{\\mathrm{NTP}} = - \\frac{1}{T_y} \\sum_{t=1}^{T_y} \\log p_\\theta(y_t \\mid y_{<t}, \\mathbf{h}_{1:T_v})$ |
| **(3)** | [`sections/3_lvr.tex` L71-L73](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L71-L73) | `equation` (unlabeled) | $\\mathcal{L} = \\mathcal{L}_{\\mathrm{NTP}} + \\lambda_{\\mathrm{LVR}} \\cdot \\mathcal{L}_{\\mathrm{LVR}}$ |
| **(4)** | [`sections/3_lvr.tex` L85-L96](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L85-L96) | `align` (unlabeled) | $J_{\\mathrm{GRPO_{latent}}}(\\theta) = \\mathbb{E}_{q, I, o \\sim \\pi_{\\theta_{\\mathrm{old}}}} \\left[ \\frac{1}{|y|} \\sum_{t=1}^{|y|} \\min \\left( r_{i,t}(\\theta) \\hat{A}_{i,t}, \\operatorname{clip}(r_{i,t}(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_{i,t} \\right) - \\beta D_{\\mathrm{KL}}(\\pi_\\theta \\| \\pi_{\\mathrm{ref}}) \\right]$ |
| **(5)** | [`sections/3_lvr.tex` L98-L100](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L98-L100) | `equation` (unlabeled) | $r_{i,t}(\\theta) = \\frac{\\pi_{\\theta}(y_{i,t}\\mid q, I, \\widetilde{h}^{\\mathrm{latent}}_{i}, y_{i,<t})}{\\pi_{\\theta_{\\mathrm{old}}}(y_{i,t}\\mid q, I, \\widetilde{h}^{\\mathrm{latent}}_{i}, y_{i,<t})}$ |
| **(6)** | [`sections/3_lvr.tex` L111-L116](file:///home/devraj/MTok/external_models/lvr/paper_src/sections/3_lvr.tex#L111-L116) | `align` (unlabeled) | $\\tilde{R}_i = \\frac{R(y_i)-\\mathrm{mean}(R(y_{1}),\\dots,R(y_{G}))}{\\mathrm{std}(R(y_{1}),\\dots,R(y_{G}))}, \\qquad \\hat{A}_{i,t} = \\tilde{R}_i$ |"""

text = text.replace(old_eq_section, new_eq_section)

# 2. Fix Issue 6: config.json
text = text.replace('`transformers_version: "4.54.0"` ([L103](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L103))',
                    '`transformers_version: "4.54.1"` ([L110](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L110))')
text = text.replace('`lvr_head_type: "mlp"` ([L32](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L32))',
                    '`lvr_head_type: "simple"` ([L17](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L17))')
text = text.replace('`loss_mode_switch_fct: "mse"` ([L26](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L26))',
                    '*(Note: `loss_mode_switch_fct` is absent from config.json)*')
text = text.replace('([L24-L35](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L24-L35))',
                    '([L13-L20](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L13-L20))')

# 3. Fix Issue 5: training_args.bin
text = text.replace('`warmup_ratio = 0.1` (`250` warmup steps)', '`warmup_ratio = 0.03` (`75` warmup steps)')

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Updated equations, config.json, and warmup_ratio successfully.")
