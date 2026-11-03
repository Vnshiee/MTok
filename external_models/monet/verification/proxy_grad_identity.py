"""
Gaps_Monet.md §1.1 says `safe_grads.append(g.detach())` + `proxy_loss = sum(v * g)` "severs" the gradient.
This loads compute_latents_only_loss verbatim from src/trainer.py and checks the identity
    d proxy / d theta  ==  (dL/dz) * (dz/dtheta)        (gradient through the latent z only)
Usage: python proxy_grad_identity.py <path to Monet repo>
"""
import ast, sys, textwrap, torch
src = open(f"{sys.argv[1]}/src/trainer.py").read()
node = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "compute_latents_only_loss"][0]
ns = {"torch": torch}; exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
torch.manual_seed(0)
W1 = torch.randn(8, 8, requires_grad=True); W2 = torch.randn(8, 8, requires_grad=True); x = torch.randn(4, 8)
z = torch.tanh(x @ W1)                      # latent produced by the model (forward 1)
L = ((torch.tanh(z @ W2) - 1) ** 2).sum()   # alignment loss downstream (forward 2), also depends on W2 directly
proxy = ns["compute_latents_only_loss"]([z], L)
gW1, gW2 = torch.autograd.grad(proxy, [W1, W2], allow_unused=True, retain_graph=True)
gz = torch.autograd.grad(L, z, retain_graph=True)[0]
ref = torch.autograd.grad(z, W1, grad_outputs=gz)[0]
print("proxy grad wrt W1 == latent-path gradient:", torch.allclose(gW1, ref, atol=1e-6), "max|diff| =", (gW1 - ref).abs().max().item())
print("||proxy grad wrt W1|| =", gW1.norm().item(), "(nonzero: not severed)")
print("proxy grad wrt W2 (direct, non-latent path) =", gW2, "(excluded, as the paper's stop_grad intends)")
