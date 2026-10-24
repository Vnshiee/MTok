Critical Review and Gap Analysis: Mull-Tokens

1. High-Level Claims and Framing

⚬ The "No Bespoke Data" Contradiction: The paper criticizes prior visual reasoning methods for depending on "bespoke task-specific training data" and handcrafted visual steps. However, the authors train Stage 1 on Zebra-CoT, an artificial, handcrafted dataset designed specifically with synthetic intermediate subgoal images (such as partially assembled jigsaw pieces). The need for bespoke data was not eliminated; it was merely shifted into pre-training.
⚬ The "World Model" Marketing: The paper claims to offer an approach "adjacent to conversations around causal world-model representations for reasoning". A true world model simulates environmental dynamics and predicts future states given an action (S_{t+1} = f(S_t, A_t)). Mull-Tokens is simply an autoregressive transformer performing standard feed-forward self-attention across 20 soft token slots. Framing a fixed-prefix tuning trick as physical world modeling overstates the architectural contribution.
⚬ The "Pareto-Dominance" Claim: The authors claim Mull-Tokens achieves "Pareto-dominance" by being both more accurate and computationally faster. This claim ignores interpretability: standard text Chain-of-Thought produces human-readable, auditable reasoning, whereas Mull-Tokens locks internal computation inside an opaque vector space. Furthermore, Mull-Tokens actually regresses compared to direct answering on specific benchmarks like Relative Depth (87.09% down to 83.87%), invalidating the claim of strict dominance across all metrics.

2. Stage 1: Warm-Up and Modality Anchoring

⚬ Smearing Spatial Information via Average-Pooling: When supervising visual thoughts, intermediate subgoal images (I_t) are passed through the image encoder and then average-pooled into a single 1D vector (v_t = \bar{g}_\phi(I_t)). Average-pooling collapses 2D spatial matrices into a single flattened coordinate, removing precise bounding boxes, object coordinates, and fine edge alignments. The claim that these tokens learn an "object layout" contradicts the mathematical reality of average-pooling.
⚬ Structural Break Between Stage 1 and Stage 2: In Stage 1, tokens are trained in a strict 1-to-1 interleaved sequence: Token 1 represents Word 1, Token 2 represents Subgoal Image 1, and so on. In Stage 2, that entire sequence structure is discarded, and the model is fed an uninterrupted block of 20 consecutive blank <Mull> tokens. The specific positional semantic roles learned during Stage 1 are completely disrupted by this sudden change in input layout.

3. Stage 2: Relaxed Mull-Tokens Training

⚬ Acknowledged Shortcut Learning: The authors explicitly admit in Section 3 that the model can bypass the latent chain entirely: "the model could learn a shortcut mapping from (x^{img}, x^{txt}) directly to y and ignore z_{1:K}, e.g., by placing most of the computation in the final layers conditioned only weakly on the <Mull> states". Because self-attention allows the answer tokens to read the input image and question directly, Stage 2 provides no architectural guarantee that the 20 <Mull> tokens are actively used.
⚬ The Static 20-Token Scaling Bottleneck: The budget of latent tokens is hard-coded to a fixed constant (K = 20). As shown in Figure 5(b), performance drops when token length is scaled to 50 or 100 tokens. In genuine reasoning, harder tasks benefit from more thinking steps; the degradation observed here indicates that the model is merely learning a static prefix vector rather than a scalable, algorithmic computation.
⚬ Catastrophic Representational Drift: Stage 2 executes 200,000 full fine-tuning iterations without any intermediate text or image loss. There are no mathematical constraints preventing the vectors from drifting away from the visual/textual grounding established in Stage 1.
⚬ A Confounding Data Distribution Shift: In Appendix 6.1, the authors reveal that the training data mix changes substantially between stages:
  ⚬ Stage 1: 40% Video-R1, 40% Zebra-CoT, 20% SAT.
  ⚬ Stage 2: 20% Video-R1, 20% Zebra-CoT, 60% SAT.
    Tripling the concentration of the SAT dataset during a 200,000-step training run confounds the evaluation. It is impossible to isolate whether gains on the SAT benchmark stem from the <Mull> mechanism or from intensive domain adaptation on the SAT distribution.
⚬ High Retraining Cost: Unlike lightweight prompt-tuning methods, Stage 2 updates all transformer layers across the 7B model for 200,000 steps using 8 \times 80 GB H100 GPUs. It is an expensive full-model retrain, not a modular plugin.

4. Stage 3: RL Refinement via GRPO

⚬ Misrepresenting Optimization as "Causal Proof": The authors claim Stage 3 "refines the causal utility of latents". Reinforcement learning optimizes mathematical correlation with a reward function, not causality. If the model discovers a superficial shortcut that correlates with a correct multiple-choice label, GRPO reinforces that shortcut. The authors performed no causal interventions (such as zeroing out latent activations during inference) to demonstrate true causal necessity.
⚬ Massive Compute for a +0.12% Net Gain: In Table 1, the overall benchmark average after Stage 2 SFT is 53.92%, while after Stage 3 GRPO it is 54.04%—a marginal increase of +0.12%. Achieving this negligible gain required 50,000 steps of DeepSpeed Stage 3 across 8 \times H100 GPUs. On several splits, such as SAT-Real, GRPO performance actually regressed (from 77.66% down to 77.00%).
⚬ A Crippled Group Size (G = 2): Standard GRPO implementations use group sizes of 8 to 16 to estimate a stable baseline. In Appendix 6.1, the authors note: "We use a group size of 2 (we run out of memory with a higher group size)". With only two sampled answers:
  ⚬ If both answers are wrong ([0, 0]), the advantage is zero; no gradient update occurs.
  ⚬ If both answers are right ([1, 1]), the advantage is zero; no gradient update occurs.
    The model only learns on [1, 0] splits, resulting in high variance and frequent dead steps.
⚬ The Credit Assignment Problem: The reward is a scalar (1 or 0) evaluated exclusively on the final answer token. This single number cannot determine which of the 20 latent tokens performed useful computation versus which added noise. The gradient washes across all 20 tokens uniformly through self-attention.
⚬ Spurious Reinforcement from Guessing: On 4-option multiple-choice benchmarks, a random guess has a 25% chance of being correct. With a group size of 2, random chance frequently produces a [1, 0] outcome, erroneously reinforcing ungrounded latent representations.

5. Empirical Results and Benchmark Claims

⚬ Cherry-Picked Headliner Metric: The abstract and introduction emphasize gains of "up to +16%". Table 1 shows this outlier comes entirely from a single sub-task: BLINK Jigsaw (58.66% \rightarrow 74.67%). Other reasoning-heavy splits show virtually no gain or outright regression:
  ⚬ BLINK IQ Test: 30.00% (Direct Answer) vs. 30.66% (Mull-Tokens + GRPO) \rightarrow a negligible +0.66%.
  ⚬ BLINK Relative Depth: 87.09% (Direct Answer) vs. 83.87% (Mull-Tokens + GRPO) \rightarrow an absolute drop of -3.22%.
⚬ Task-Specific Data Leakage: The large gain on BLINK Jigsaw (+16.01%) directly corresponds to the composition of the Stage 1 warm-up data. Figure 3 explicitly confirms that Zebra-CoT contains step-by-step jigsaw puzzle tasks ("A jigsaw puzzle shows a picture of pandas but has several missing pieces. Let's think by fitting first image..."). This improvement reflects targeted domain exposure rather than general visual reasoning.
⚬ Near-Zero Multi-Image Generalization: Table 3 presents generalization on MMSI-Bench, yet the overall average moves from 28.8% (Direct Answer) to 29.0% (Mull-Tokens). An improvement of +0.2% falls within standard evaluation noise and does not support the claim of broader generalization.
⚬ Missing Error Bars and Significance Testing: None of the evaluation tables (Tables 1, 2, 3, 4, 5) provide standard deviations (\pm \sigma) or confidence intervals across multiple random seeds.

6. Ablation Studies and Latent Dynamics

⚬ Multimodal Warm-Up Regresses Against Text Warm-Up: The central premise of the paper is that warm-up tokens must be multimodal. In Table 2, however, <Mull> trained with Text-only warm-up outperforms Multimodal warm-up on two out of the four evaluated benchmark columns:
  ⚬ VSI-Bench Reasoning: Text warm-up scores 33.5%, while Multimodal warm-up drops to 32.9%.
  ⚬ ERQA: Text warm-up scores 38.5%, while Multimodal warm-up drops to 38.3%.
    Almost all of the reported multimodal benefit is isolated to a single benchmark (SAT-Real: 71.3% \rightarrow 77.7%).
⚬ The Authors' Parenthetical Admission on Latent Utility: In Appendix 6.2, the authors evaluate whether the latent tokens avoid representational collapse by measuring Euclidean distance between successive tokens. They include the following admission:
  "This indicates that the model avoids representational collapse and that each latent continues to contribute information (albeit useful or not)."
  The authors explicitly acknowledge that non-identical vector states do not prove the tokens are performing valid reasoning rather than propagating arbitrary numerical drift.
⚬ The Interpretability Tax: In Table 4, when the model is trained to generate human-readable text rationales after the <Mull> tokens, overall accuracy drops from 53.92% down to 51.10%. Forcing the model to explain its thoughts reintroduces the very performance penalties Mull-Tokens was designed to avoid.

7. Experimental Setup and Evaluation Biases

⚬ Artificially Degraded Text Baselines: In Appendix 6.1, the system prompt used for the Text-Reasoning Baseline explicitly forces the model to generate conversational filler:
  "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc..."
  In contrast, the prompt for Mull-Tokens removes all filler requests:
  "Please think about this question deeply... Provide your final answer between the <answer> </answer> tags."
  Forcing the text baseline to produce conversational padding encourages hallucination and visual drift. The text baseline was handicapped by its prompt design.
⚬ Fragile Regex Extraction Penalizes Long Text Outputs: In Appendix 6.1, the answer parsing pipeline employs a fallback rule: if no XML tags or standard phrases match, it extracts the last isolated capital letter in the output text. In a 400-word Chain-of-Thought response evaluating multiple options sequentially, the final capital letter mentioned is frequently an eliminated option rather than the final selection. The short outputs produced by Mull-Tokens (<answer> B </answer>) avoid this extraction failure entirely, skewing parsing accuracy in favor of Mull-Tokens.
⚬ The Frozen Vision Encoder Limitation: In Appendix 6.1, the authors state: "We freeze the vision encoder... All other parts of the model are fully fine-tuned". Because the vision backbone is frozen, the model cannot update its low-level feature extraction to capture fine spatial boundaries or depth cues. This explains why fine-grained visual perception tasks like Relative Depth show negative transfer.
⚬ Narrow Model Scale Evaluation: The experiments are conducted exclusively on a single 7B parameter model (Qwen2.5-VL 7B). The paper provides no empirical evidence that modality-agnostic latent tokens remain beneficial on larger architectures (such as 32B or 72B models), where native text reasoning exhibits significantly less visual drift.