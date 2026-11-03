"""
VLPO faithfulness test for Monet (repo NOVAglow646/Monet @ 0893999).

Rollout: a tiny random Qwen2 decoder is decoded token by token. The latent state machine is the
repo's own code, executed verbatim from RL/monet_models/vllm/monet_gpu_model_runner.py L1305-L1339
(activate on <latent>, record last-position hidden state as the next input, force </latent> after K).
The input override follows L1099-L1121: the new token's row gets the pending hidden state.

Actor: the repo's own collect_varlen_segment_indices and compute_latent_log_probs
(RL/verl/workers/actor/dp_actor.py), and the insertion line `inputs_embeds[0, latent_poss] = latents`
(RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py L1858), on the same weights (on-policy).

Questions:
  Q1  Does the actor's training context reproduce the rollout context? (text-token log-probs)
  Q2  On-policy, what latent log-prob does the repo code assign? (0 if the Gaussian mean is the latent's own producer)
  Q3  Gradient of the VLPO latent term at theta = theta_old: repo code vs. a correctly aligned mean.
  Q4  What happens when ABS_VIS_START_ID / ABS_VIS_END_ID are unset (never set in the repo)?
"""
import ast, json, os, sys, textwrap, types
import torch, torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

REPO = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(SEED)

def func_src(path, name):
    src = open(f"{REPO}/{path}").read()
    node = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name][0]
    return textwrap.dedent(ast.get_source_segment(src, node))

def lines(path, a, b):
    return "".join(open(f"{REPO}/{path}").readlines()[a - 1:b])

ns = {"torch": torch, "os": os}
exec(func_src("RL/verl/workers/actor/dp_actor.py", "collect_varlen_segment_indices"), ns)
exec(func_src("RL/verl/workers/actor/dp_actor.py", "compute_latent_log_probs"), ns)
collect, latent_logp = ns["collect_varlen_segment_indices"], ns["compute_latent_log_probs"]

runner_block = textwrap.dedent(lines("RL/monet_models/vllm/monet_gpu_model_runner.py", 1305, 1339))
assert "sampled = mu  # + torch.randn_like(mu)" in runner_block, "rollout noise line changed"
assert 'st["current_len"] >= self.latent_size' in runner_block
insert_line = lines("RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py", 1856, 1858)
assert "inputs_embeds[0, latent_poss] = latents" in insert_line
actor_env_block = lines("RL/verl/workers/actor/dp_actor.py", 195, 196)
assert 'int(os.getenv("ABS_VIS_START_ID"))' in actor_env_block

V, H = 96, 64
cfg = Qwen2Config(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=int(os.getenv("LAYERS", "2")), num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=False,
                  initializer_range=float(os.getenv("INIT", "0.02")), rms_norm_eps=1e-6)
model = Qwen2ForCausalLM(cfg).eval()
START, END = 90, 91
P, NEW = 10, K + 12

# ------------------------------- rollout ------------------------------------
runner = types.SimpleNamespace(latent_state={}, latent_start_id=START, latent_end_id=END, latent_size=K,
                               input_batch=types.SimpleNamespace(req_ids=["r0"]))
prompt = torch.randint(0, 80, (1, P))
emb = model.model.embed_tokens
ids = prompt.clone()
with torch.no_grad():
    o = model.model(inputs_embeds=emb(prompt), use_cache=True)
    cache, h_last = o.past_key_values, o.last_hidden_state[:, -1]
    roll_logits, recorded, input_vec = [], [], {}
    for step in range(NEW):
        logits = model.lm_head(h_last)
        roll_logits.append(logits[0])
        tid = START if step == 1 else int(logits[0, :80].argmax())     # force a <latent> at step 1, else greedy text
        valid_sampled_token_ids = [[tid]]
        latents_step = [None]
        loc = dict(self=runner, valid_sampled_token_ids=valid_sampled_token_ids, last_token_h=h_last,
                   latents_step=latents_step, os=os, torch=torch)
        exec("for i, req_id in enumerate(self.input_batch.req_ids):\n" + textwrap.indent(
            runner_block.split("\n", 1)[1], ""), loc)                  # verbatim body of the per-request loop
        if latents_step[0] is not None:
            recorded.append(latents_step[0])
        new_tid = valid_sampled_token_ids[0][0]
        ids = torch.cat([ids, torch.tensor([[new_tid]])], 1)
        pos = ids.shape[1] - 1
        st = runner.latent_state["r0"]
        e = emb(torch.tensor([[new_tid]]))
        if st.get("active") and st.get("pending") is not None:          # L1107-L1116: override the new token's row
            e = st["pending"].to(e.dtype).view(1, 1, -1); input_vec[pos] = st["pending"].float().clone()
            st["pending"] = None                                        # L1118-L1121
        o = model.model(inputs_embeds=e, past_key_values=cache, use_cache=True)
        cache, h_last = o.past_key_values, o.last_hidden_state[:, -1]
roll_logp = torch.stack([F.log_softmax(l.float(), -1) for l in roll_logits])   # predicts ids[P + s]
resp = ids[0, P:]
s_pos = int((ids[0] == START).nonzero()[0]); e_pos = int((ids[0] == END).nonzero()[0])
overridden = sorted(input_vec)
latents = torch.stack(recorded).float()            # what the recorder returns (fp16 on the wire)

# ------------------------------- actor --------------------------------------
_, by_batch = collect(input_ids=ids, attention_mask=torch.ones_like(ids), start_id=START, end_id=END)
latent_poss = torch.cat(by_batch[0])

def actor_forward(insert=True, params_grad=False):
    inputs_embeds = emb(ids).clone()
    lp, lt = (latent_poss, latents) if insert else (None, None)
    loc = dict(inputs_embeds=inputs_embeds, latent_poss=lp, latents=lt)
    exec(textwrap.dedent(insert_line), loc)                    # verbatim insertion rule
    out = model.model(inputs_embeds=loc["inputs_embeds"])
    return out.last_hidden_state

with torch.no_grad():
    hs = actor_forward()
    hs_noins = actor_forward(insert=False)
def text_mismatch(hidden):
    lp = F.log_softmax(model.lm_head(hidden[0]).float(), -1)
    errs = []
    for s in range(len(resp)):
        q = P + s
        if s_pos <= q <= e_pos and q != e_pos:           # tokens inside the latent segment are not text actions
            continue
        errs.append(abs(lp[q - 1, ids[0, q]] - roll_logp[s, ids[0, q]]).item())
    return max(errs)

repo_lp = latent_logp(latent_poss, latents, hs, sigma=10.0)               # repo: mean = h at latent_poss
fixed_lp = latent_logp(latent_poss - 1, latents, hs, sigma=10.0)          # mean = h at latent_poss-1 (producer)

# Q3 gradient at theta_old of sum(advantage * latent_logp), advantage = +1
def grad_norm(shift):
    model.zero_grad()
    hs_g = actor_forward()
    lp = latent_logp(latent_poss - shift, latents, hs_g.float(), sigma=10.0)
    lp.sum().backward()
    return torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
g_repo, g_fixed = grad_norm(0), grad_norm(1)
model.zero_grad()

# Q4 env-var fallback: run the actor's first lines exactly as in the repo with the vars unset
os.environ.pop("ABS_VIS_START_ID", None); os.environ.pop("ABS_VIS_END_ID", None)
try:
    exec(textwrap.dedent(actor_env_block), {"os": os}); env_ok = True
except Exception as ex:
    env_ok = f"{type(ex).__name__}: {ex}"

cos = lambda a, b: F.cosine_similarity(a, b, dim=-1)
res = {
  "K": K, "seed": SEED,
  "<latent>_pos": s_pos, "</latent>_pos": e_pos,
  "rollout_overridden_input_positions": overridden,
  "actor_latent_poss": latent_poss.tolist(),
  "n_recorded_latents": len(recorded),
  "Q1 max|dlogp| text tokens, actor(with latents) vs rollout": text_mismatch(hs),
  "Q1 max|dlogp| text tokens, actor(no latents inserted) vs rollout": text_mismatch(hs_noins),
  "Q2 repo latent logp (sigma=10), on-policy": [round(x, 4) for x in repo_lp.tolist()],
  "Q2 aligned latent logp (mean=h[poss-1])": [round(x, 6) for x in fixed_lp.tolist()],
  "Q2 cos(latent_j, h_repo_mean_j)": [round(x, 3) for x in cos(latents, hs[0, latent_poss]).tolist()],
  "Q2 cos(latent_j+1, h_repo_mean_j) (repo mean == next latent?)": [round(x, 3) for x in cos(latents[1:], hs[0, latent_poss[:-1]]).tolist()],
  "Q3 ||grad|| of VLPO latent term at theta_old: repo": g_repo,
  "Q3 ||grad|| ... aligned mean (fp16-stored latents)": g_fixed,
  "Q4 actor env block with ABS_VIS_* unset": env_ok,
}
print(json.dumps(res, indent=1))
