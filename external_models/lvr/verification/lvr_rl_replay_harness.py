"""
Faithfulness test for LVR's GRPO_latent replay (repo VincentLeebang/lvr @ 637a9f7).

Question: does the RL loss forward condition the answer tokens on the SAME latent
context that the rollout produced (paper 3_lvr.tex L97-L106: "restoring the exact
context preceding text generation")?

Method: a tiny random Qwen2 decoder gets the repo's LVR forward semantics
(monkey_patch_forward_lvr_rl.py L129-L142, copied verbatim) and the repo's own
generation loop (qwen_lvr_model.py: generate + _lvr_deocding_by_steps) and replay
functions (grpo_trainer.py: _compute_lvr_spans, _replay_lvr_to_collect_states,
score_with_lvr_replay, the L742-L757 mask loop and the L996-L1034 loss forward),
all loaded from the repo source text. We then compare per-token log-probs of the
generated completion under (a) the rollout itself and (b) each replay path.
"""
import ast, sys, textwrap, types, math, json
import torch, torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig, LogitsProcessor, LogitsProcessorList
from transformers.utils import ModelOutput
from dataclasses import dataclass

REPO = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(SEED)
sys.path.insert(0, REPO)

def src_lines(path, a, b):
    with open(f"{REPO}/{path}") as f:
        return "".join(f.readlines()[a - 1:b])

def method_src(path, name):
    tree = ast.parse(open(f"{REPO}/{path}").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(open(f"{REPO}/{path}").read(), node)
            # keep decorators (e.g. @torch.no_grad())
            deco = "".join("@" + ast.get_source_segment(open(f"{REPO}/{path}").read(), d) + "\n" for d in node.decorator_list)
            return deco + textwrap.dedent(seg)
    raise KeyError(name)

# ---- repo generation loop (verbatim) ----
from src.model import qwen_lvr_model as QLM
gen_fn = QLM.QwenWithLVR.generate
steps_fn = QLM.QwenWithLVR._lvr_deocding_by_steps

# ---- repo trainer functions (verbatim source) ----
ns = {"torch": torch, "F": F, "Tuple": Tuple, "Optional": Optional, "List": List}
for name in ["_compute_lvr_spans", "_replay_lvr_to_collect_states", "score_with_lvr_replay"]:
    exec(method_src("src/trainer/grpo_trainer.py", name), ns)
mask_block = textwrap.dedent(src_lines("src/trainer/grpo_trainer.py", 742, 757))
assert "lvr_mask[b, t] = False" in mask_block and "final_mask = attention_mask.bool() & lvr_mask" in mask_block
loss_fwd_block = textwrap.dedent(src_lines("src/trainer/grpo_trainer.py", 996, 1034))
assert "_replay_lvr_to_collect_states" in loss_fwd_block and "selective_log_softmax" in loss_fwd_block
rl_patch_block = textwrap.dedent(src_lines("src/train/monkey_patch_forward_lvr_rl.py", 129, 142))
assert "inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]" in rl_patch_block
assert "comp_embeds = torch.where(" in rl_patch_block

@dataclass
class Out(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: Optional[object] = None
    last_position_hidden_state: Optional[torch.FloatTensor] = None

class ToyLVR(Qwen2ForCausalLM):
    generate = gen_fn
    _lvr_deocding_by_steps = steps_fn
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, cache_position=None, lvr_mode_switch=None,
                last_position_hidden_state=None, lvr_states=None, lvr_mask=None, prompt_length=None,
                return_dict=True, logits_to_keep=None, **kw):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids).clone()
        loc = dict(inputs_embeds=inputs_embeds, lvr_mode_switch=lvr_mode_switch,
                   last_position_hidden_state=last_position_hidden_state, lvr_states=lvr_states,
                   lvr_mask=lvr_mask, prompt_length=prompt_length, torch=torch)
        exec(rl_patch_block, loc)            # repo's exact patching rules
        inputs_embeds = loc["inputs_embeds"]
        o = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
                       past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position)
        h = o.last_hidden_state
        return Out(logits=self.lm_head(h), past_key_values=o.past_key_values, last_position_hidden_state=h[:, -1, :])

V, H = 96, 64
cfg = Qwen2Config(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  tie_word_embeddings=False, use_cache=True)
model = ToyLVR(cfg).eval()
LS, LE, LVR_ID, EOS = 90, 91, 92, 95
model.config.lvr_start_id, model.config.lvr_end_id, model.config.lvr_id = LS, LE, LVR_ID
model.config.is_encoder_decoder = False
model.config.lvr_head = False

class Force(LogitsProcessor):
    """Force LS as 1st completion token and LE right after the K-step quota, else free (greedy)."""
    def __init__(self, P, K): self.P, self.K = P, K
    def __call__(self, ids, scores):
        c = ids.shape[1] - self.P
        s = scores.clone()
        if c == 0: s[:] = -1e9; s[:, LS] = 0
        elif c == self.K + 1: s[:] = -1e9; s[:, LE] = 0
        else: s[:, [LS, LE, LVR_ID, EOS]] = -1e9
        return s

B, P, NEW = 3, 12, K + 14
prompt = torch.randint(0, 80, (B, P))
gc = GenerationConfig(max_new_tokens=NEW, do_sample=False, output_logits=True, return_dict_in_generate=True,
                      pad_token_id=0, eos_token_id=EOS)
gc.lvr_steps = K
with torch.no_grad():
    out = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=gc,
                         logits_processor=LogitsProcessorList([Force(P, K)]),
                         decoding_strategy="steps", lvr_steps=[K] * B)
seq = out.sequences
comp = seq[:, P:]
C = comp.shape[1]
roll_logp = torch.stack([F.log_softmax(l.float(), -1) for l in out.logits], 1)   # (B, C, V)
roll_tok_logp = roll_logp.gather(-1, comp.unsqueeze(-1)).squeeze(-1)

# ---- trainer-side objects -----------------------------------------------------
trainer = types.SimpleNamespace(model=model, generation_config=gc, temperature=1.0)
for name in ["_compute_lvr_spans", "_replay_lvr_to_collect_states", "score_with_lvr_replay"]:
    setattr(trainer, name, types.MethodType(ns[name], trainer))

prompt_completion_ids = seq
prompt_mask = torch.ones_like(prompt)
completion_mask = torch.ones_like(comp)
loc = dict(torch=torch, prompt_completion_ids=prompt_completion_ids, self=trainer,
           prompt_mask=prompt_mask, completion_mask=completion_mask)
exec(mask_block, loc)
final_mask = loc["final_mask"][:, -C:]

def selective_log_softmax(logits, idx):
    return F.log_softmax(logits.float(), -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)

# (1) repo loss-forward path (no_grad replay + patched teacher-forcing forward), verbatim L996-L1034
loc = dict(self=trainer, model=model, prompt_ids=prompt, prompt_mask=prompt_mask,
           prompt_completion_ids=prompt_completion_ids, multimodal_inputs={}, logits_to_keep=C,
           selective_log_softmax=selective_log_softmax, torch=torch)
with torch.no_grad():
    exec(loss_fwd_block, loc)
loss_path_logp = loc["per_token_logps"]
lvr_states, lvr_mask_rep = loc["lvr_states"], loc["lvr_mask"]

# (2) repo score_with_lvr_replay path (used for old/ref log-probs)
with torch.no_grad():
    srep = trainer.score_with_lvr_replay(model, input_ids=prompt, attention_mask=prompt_mask, target_ids=seq,
                                         generation_config=gc, lvr_steps=[K] * B, enable_grad=False)

# (3) corrected replay: patch input position j with h_{j-1} from a faithful replay of the rollout
def corrected_states():
    st = torch.zeros(B, C, H); m = torch.zeros(B, C, dtype=torch.bool)
    with torch.no_grad():
        start = (comp == LS).float().argmax(1)
        cache = None; inp = seq[:, :P]; lphs = None; mode = torch.zeros(B, dtype=torch.bool)
        o = model(input_ids=inp, use_cache=True); cache = o.past_key_values; lphs = o.last_position_hidden_state
        for c in range(C - 1):
            pos = P + c
            sw = (c > start) & (c <= start + K)         # input positions start+1..start+K carry h_{pos-1}
            if sw.any():
                st[sw, c] = lphs[sw]; m[sw, c] = True
            o = model(input_ids=seq[:, pos:pos + 1], past_key_values=cache, use_cache=True,
                      lvr_mode_switch=sw, last_position_hidden_state=lphs if sw.any() else None,
                      cache_position=torch.tensor([pos]))
            cache, lphs = o.past_key_values, o.last_position_hidden_state
    return st, m
cst, cm = corrected_states()
with torch.no_grad():
    lo = model(input_ids=seq, lvr_states=cst, lvr_mask=cm, prompt_length=P).logits[:, :-1][:, -C:]
corr_logp = selective_log_softmax(lo, seq[:, 1:][:, -C:])

# ---- report --------------------------------------------------------------------
fm = final_mask.bool()
def maxdiff(a): return (a - roll_tok_logp).abs()[fm].max().item()
lat_pos_rep = [torch.nonzero(lvr_mask_rep[b]).flatten().tolist() for b in range(B)]
lat_pos_cor = [torch.nonzero(cm[b]).flatten().tolist() for b in range(B)]
ls_idx = (comp == LS).float().argmax(1).tolist()
ls_in_loss = [bool(final_mask[b, ls_idx[b]]) for b in range(B)]
le_idx = (comp == LE).float().argmax(1).tolist()
res = {
  "K": K, "seed": SEED, "completion_len": C,
  "LS_completion_index": ls_idx, "LE_completion_index": le_idx,
  "rollout_latent_input_positions(start+1..start+K)": [list(range(s + 1, s + K + 1)) for s in ls_idx],
  "repo_replay_patched_positions": lat_pos_rep,
  "corrected_patched_positions": lat_pos_cor,
  "lvr_start_token_included_in_policy_loss": ls_in_loss,
  "n_tokens_in_loss_per_sample": fm.sum(1).tolist(),
  "max|logp_repo_loss_forward - logp_rollout| (loss tokens)": maxdiff(loss_path_logp),
  "max|logp_score_with_lvr_replay - logp_rollout| (loss tokens)": maxdiff(srep[:, -C:]),
  "max|logp_corrected_replay - logp_rollout| (loss tokens)": maxdiff(corr_logp),
}
print(json.dumps(res, indent=1))

# ---- diagnostics: which state does the repo patch at each latent position? ----
with torch.no_grad():
    full = model(input_ids=seq[:, :P + 1])           # prompt + LS, no patching
    h_before_LS = model(input_ids=seq[:, :P]).last_position_hidden_state   # h_{P-1}
b0 = 0; s0 = ls_idx[b0]
def cos(a, b): return F.cosine_similarity(a, b, dim=-1).item()
diag = {
 "repo_state@first_latent_pos == h(position before LS)": torch.allclose(lvr_states[b0, s0 + 1].float(), h_before_LS[b0], atol=1e-5),
 "repo_state@first_latent_pos == rollout state (h at LS)": torch.allclose(lvr_states[b0, s0 + 1].float(), cst[b0, s0 + 1], atol=1e-5),
 "cos(repo_state, rollout_state) per latent position": [round(cos(lvr_states[b0, j].float(), cst[b0, j]), 4) for j in range(s0 + 1, s0 + K + 1)],
 "cos(repo_state[j], rollout_state[j-1]) j>=2": [round(cos(lvr_states[b0, j].float(), cst[b0, j - 1]), 4) for j in range(s0 + 2, s0 + K + 1)],
}
# temperature mismatch: ref/old scored at apply_temperature=1.0 (default, never overridden),
# policy loss logits divided by self.temperature (=0.9 in finetune_lvr_stage2_*.sh) -> KL at policy==ref
T = 0.9
with torch.no_grad():
    lo_pol = model(input_ids=seq, lvr_states=cst, lvr_mask=cm, prompt_length=P).logits[:, :-1][:, -C:] / T
pol = selective_log_softmax(lo_pol, seq[:, 1:][:, -C:])
ref = srep[:, -C:]
kl = torch.exp(ref - pol) - (ref - pol) - 1        # repo L1072 formula
diag["mean per-token KL at init (policy==ref), T_policy=0.9 vs T_ref=1.0"] = (kl * final_mask).sum().item() / final_mask.sum().item()
print(json.dumps(diag))

# ---- proposed minimal fix to _replay_lvr_to_collect_states, verified here ----
fixed_src = method_src("src/trainer/grpo_trainer.py", "_replay_lvr_to_collect_states")
old_switch = '''                    "lvr_mode_switch": in_span, '''
assert old_switch in fixed_src
# (a) the forward at this step processes position pos-1, so gate it on pos-1 being a latent input position
fixed_src = fixed_src.replace(old_switch, '''                    "lvr_mode_switch": ((pos - 1 > lvr_start_idx) & (pos - 1 <= configured_end_idx)).to(device=device), ''')
# (b) the state fed into input position pos is the output at pos-1, i.e. THIS step's output: collect after the forward
old_collect = fixed_src[fixed_src.index("            if update_mask.any():"):fixed_src.index("            # prepare model inputs for this step")]
fixed_src = fixed_src.replace(old_collect, "")
anchor = "            last_position_hidden_state = getattr(outputs, \"last_position_hidden_state\", last_position_hidden_state)\n"
assert anchor in fixed_src
fixed_src = fixed_src.replace(anchor, anchor + old_collect)
ns2 = dict(ns); exec(fixed_src, ns2)
trainer.__dict__["_replay_lvr_to_collect_states"] = types.MethodType(ns2["_replay_lvr_to_collect_states"], trainer)
loc = dict(self=trainer, model=model, prompt_ids=prompt, prompt_mask=prompt_mask,
           prompt_completion_ids=prompt_completion_ids, multimodal_inputs={}, logits_to_keep=C,
           selective_log_softmax=selective_log_softmax, torch=torch)
with torch.no_grad():
    exec(loss_fwd_block, loc)
print(json.dumps({"FIXED repo loss-forward: max|logp - logp_rollout|": maxdiff(loc["per_token_logps"])}))
