import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

out_dir = '/home/devraj/MTok/slides/figures'

# Configure global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'text.color': '#1e293b',
})

# -------------------------------------------------------------
# 8. Slide 7: Standard VLM Pipeline & Bottlenecks
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# Boxes
boxes = [
    (0.8, 2.0, 'Input\nImage', '#f1f5f9', '#64748b'),
    (2.3, 2.0, 'ViT\nEncoder', '#e0f2fe', '#0284c7'),
    (3.8, 2.0, 'Linear\nProjector', '#e0e7ff', '#6366f1'),
    (5.4, 2.0, 'Visual Tokens\n(100s–1000s)\n⊕ Text Tokens', '#fef3c7', '#d97706'),
    (7.5, 2.0, 'LLM Decoder\n(Fixed 28 Layers)\nAutoregressive', '#fce7f3', '#db2777'),
    (9.2, 2.0, 'Final\nAnswer', '#dcfce7', '#10b981'),
]

for xc, yc, txt, fc, ec in boxes:
    ax.text(xc, yc, txt, ha='center', va='center', fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor=fc, edgecolor=ec, lw=1.5))

# Arrows
arrows = [(1.3, 1.8), (2.8, 3.3), (4.4, 4.6), (6.3, 6.7), (8.3, 8.8)]
for x1, x2 in arrows:
    ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.8, color='#475569'))

# Bottleneck Annotations
# 1. Above ViT
ax.annotate('Bottleneck 1: Visual enters ONCE\n(Frozen in most recipes; no re-attending)',
            xy=(2.3, 2.65), xytext=(2.3, 3.4),
            ha='center', fontsize=7.5, fontweight='bold', color='#0284c7',
            arrowprops=dict(arrowstyle="->", lw=1.2, color='#0284c7'))

# 2. Above Projector / Tokens
ax.annotate('Bottleneck 2: Thousands of visual tokens\n(High spatial dilution across prompt)',
            xy=(5.4, 2.8), xytext=(5.4, 3.5),
            ha='center', fontsize=7.5, fontweight='bold', color='#d97706',
            arrowprops=dict(arrowstyle="->", lw=1.2, color='#d97706'))

# 3. Above LLM Decoder
ax.annotate('Bottleneck 3: Fixed Depth (28 Layers)\n(Only way to compute more is emitting tokens)',
            xy=(7.5, 2.85), xytext=(7.5, 3.5),
            ha='center', fontsize=7.5, fontweight='bold', color='#db2777',
            arrowprops=dict(arrowstyle="->", lw=1.2, color='#db2777'))

# Bottom takeaway
ax.text(5.0, 0.45, 'Core Tension: Emitting tokens forces commitment to words (Text CoT),\nwhich actively hurts visual-spatial manipulation.',
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='#991b1b',
        bbox=dict(boxstyle='square,pad=0.4', facecolor='#fee2e2', edgecolor='#ef4444', lw=1))

plt.title('The Standard Vision-Language Architecture and Its Inherent Constraints', fontsize=11, fontweight='bold', pad=12)
plt.savefig(f'{out_dir}/slide7_standard_pipeline.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 9. Slide 9: The Investigation Funnel
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.3), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Funnel trapezoids / stages
stages = [
    (4.3, '1. Models fail at spatial manipulation, not perception', '#e2e8f0', '#64748b', 9.0),
    (3.3, '2. Verbalising manipulation makes performance worse', '#fee2e2', '#ef4444', 7.5),
    (2.3, '3. Give model computation it does not verbalise → Latent Tokens', '#e0f2fe', '#0284c7', 6.2),
    (1.2, '4. Crucial Question: Does latent computation actually happen?', '#fef3c7', '#d97706', 5.0),
]

for y, text, fc, ec, width in stages:
    ax.text(5.0, y, text, ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=fc, edgecolor=ec, lw=1.5))
    if y > 1.5:
        ax.annotate('', xy=(5.0, y-0.45), xytext=(5.0, y-0.2),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748b'))

ax.text(5.0, 0.25, 'Every method reports accuracy gains. Almost none audits what latents do.',
        ha='center', va='center', fontsize=9, fontweight='bold', color='#1e293b')

plt.title('Narrowing to the Core Scientific Question', fontsize=11, fontweight='bold', pad=10)
plt.savefig(f'{out_dir}/slide9_funnel.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 10. Slide 11: Mull-Tokens 3-Stage Training Pipeline
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)
ax.axis('off')

# 3 Stages horizontal flow
stg1 = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#0284c7', lw=1.5)
stg2 = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#10b981', lw=1.5)
stg3 = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#db2777', lw=1.5)

ax.text(1.8, 2.4, "Stage 1: Warm-up\n\n• Interleaved latents\n  (1 latent per word)\n• Next-token cross-entropy\n• Cosine loss to pooled\n  subgoal image features",
        bbox=stg1, ha='center', va='center', fontsize=8, color='#1e293b')

ax.text(5.0, 2.4, "Stage 2: Relaxed SFT\n\n• 20 Contiguous Latents\n  (<|latent_pad|> x 20)\n• Zero intermediate loss\n• Answer loss only\n• Unsupervised latents",
        bbox=stg2, ha='center', va='center', fontsize=8, color='#1e293b')

ax.text(8.2, 2.4, "Stage 3: GRPO RL\n\n• Reinforcement learning\n• Reward: Answer correct\n• Group size G=2\n• Code audit: policy loss on\n  text_only_logprobs=True",
        bbox=stg3, ha='center', va='center', fontsize=8, color='#1e293b')

# Arrows
ax.annotate('', xy=(3.3, 2.4), xytext=(3.0, 2.4), arrowprops=dict(arrowstyle="->", lw=2, color='#64748b'))
ax.annotate('', xy=(6.5, 2.4), xytext=(6.2, 2.4), arrowprops=dict(arrowstyle="->", lw=2, color='#64748b'))

# Callouts at bottom
ax.text(3.4, 0.5, "Structural Mismatch: Interleaved in Stg 1 vs. 20 Contiguous in Stg 2",
        ha='center', va='center', fontsize=8, fontweight='bold', color='#0369a1',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0f2fe', edgecolor='#0284c7'))

ax.text(8.2, 0.5, "GRPO Latent Disconnect:\nLatents receive no direct policy gradients",
        ha='center', va='center', fontsize=8, fontweight='bold', color='#991b1b',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#ef4444'))

plt.title('Mull-Tokens 3-Stage Training Pipeline (Ray et al., CVPR 2026)', fontsize=11, fontweight='bold', pad=12)
plt.savefig(f'{out_dir}/slide11_mull_pipeline.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 11. Slide 18: Master Empirical Architecture (E1 & E2)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.4)
ax.axis('off')

# Flowchart boxes
ax.text(5.0, 3.9, "Initial Gap Analysis & Mechanistic Hypotheses (FinalGaps.md)",
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f1f5f9', edgecolor='#64748b', lw=1.2))

ax.annotate('', xy=(5.0, 3.2), xytext=(5.0, 3.6), arrowprops=dict(arrowstyle="->", lw=1.8, color='#475569'))

ax.text(5.0, 2.5, "Experiment E1: Mechanistic Factorial Matrix\nN = 10,500 evaluations | 15 conditions | Standardized 4.19 MP\nTarget: Scratchpad Compute (H1) vs. Format Stop-Gate (H3)",
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0369a1',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#e0f2fe', edgecolor='#0284c7', lw=1.5))

ax.text(5.0, 1.6, "Key Finding: <think></think> matches 20 latents (p = 0.5413); K=0 collapse is verbal runaway\nDiscovery: 10-point 'B' prior matches Multi-View test imbalance",
        ha='center', va='center', fontsize=8, fontweight='bold', color='#991b1b',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#ef4444', lw=1))

ax.annotate('', xy=(5.0, 0.95), xytext=(5.0, 1.25), arrowprops=dict(arrowstyle="->", lw=1.8, color='#475569'))

ax.text(5.0, 0.5, "Experiment E2: Bias vs. Discrimination Study\nN = 44,590 evaluations | 14 conditions | 6 input variants (V0–V4, CF)\nTarget: Disentangle true vision (s) from letter (L) & position (P) priors",
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='#065f46',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#dcfce7', edgecolor='#10b981', lw=1.5))

plt.title('Master Empirical Investigation Architecture', fontsize=11, fontweight='bold', pad=12)
plt.savefig(f'{out_dir}/slide18_empirical_architecture.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 12. Slide 21: Decoded Text Pathology Comparison
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2), dpi=300)

# Left: Row 1
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.text(5.0, 9.2, "Row 1: Format-Locked (20 Latents)", ha='center', fontsize=9.5, fontweight='bold', color='#0369a1')

r1_text = (
    "Input Prefill:\n"
    "  <think> <|latent_pad|> x 20 </think>\\n\n\n"
    "Generated Output Tokens:\n"
    "  <answer>B</answer><|im_end|>\n\n"
    "Characteristics:\n"
    "  • Exactly 7.0 tokens across ALL 700 items\n"
    "  • Variance σ = 0.00\n"
    "  • Mean Latency: 0.36 seconds / sample\n"
    "  • Regex extraction: 100% clean parse"
)
ax1.text(0.5, 4.5, r1_text, fontsize=8, family='monospace', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f9ff', edgecolor='#0284c7', lw=1.2))

# Right: Row 2
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.text(5.0, 9.2, "Row 2: Unlocked Empty Prefill (K=0)", ha='center', fontsize=9.5, fontweight='bold', color='#991b1b')

r2_text = (
    "Input Prefill:\n"
    "  (empty assistant turn)\n\n"
    "Generated Output Tokens:\n"
    "  <think>THOUGHT: The red object is left of...\n"
    "  wait, looking closer at point 2...\n"
    "  repeating: 'The 48th image is a larger...'\n\n"
    "Characteristics:\n"
    "  • Mean tokens: 138.5 (Max: 1,024 cap)\n"
    "  • Mean Latency: 175.12 seconds / sample\n"
    "  • Catastrophic repetition loops (IQ #52)\n"
    "  • 3 hard regex parse crashes"
)
ax2.text(0.5, 4.5, r2_text, fontsize=8, family='monospace', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef2f2', edgecolor='#ef4444', lw=1.2))

plt.suptitle('Decoding Pathology: Why Row 2 ($K=0$) Collapsed by 12.00%', fontsize=11, fontweight='bold')
plt.savefig(f'{out_dir}/slide21_decoding_pathology.pdf', bbox_inches='tight')
plt.close()

print('All supplementary figures generated successfully!')
