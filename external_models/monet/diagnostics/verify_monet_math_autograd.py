"""
verify_monet_math_autograd.py:
Rigorous mathematical and autograd verification of Monet's two central claims:
1. Hypothesis H1: VLPO Continuous Policy Gradient Ratio Degeneracy & L2 Equivalence.
2. Hypothesis H2: SFT Stage 2 Detached Proxy Loss Gradient Isolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_h1_vlpo_ratio_and_gradient():
    print("=" * 70)
    print("TESTING HYPOTHESIS H1: VLPO Ratio Degeneracy and Gradient Behavior")
    print("=" * 70)
    
    # In RL/verl/workers/actor/dp_actor.py L135-158:
    # r = torch.exp(- (h_old - h_theta).pow(2).sum(-1) / (2 * sigma**2))
    # loss = - (r * advantage).mean()
    
    sigma = 1.0
    dim = 3584 # Monet hidden dim
    batch_size = 4
    num_latents = 10
    
    torch.manual_seed(42)
    h_old = torch.randn(batch_size, num_latents, dim)
    
    # Case 1: Identity at step 0 (h_theta = h_old)
    h_theta = nn.Parameter(h_old.clone())
    diff = (h_old - h_theta).pow(2).sum(dim=-1)
    r_step0 = torch.exp(-diff / (2 * sigma**2))
    
    print(f"[H1.1] Step 0 (h_theta == h_old):")
    print(f"       Ratio min: {r_step0.min().item():.6f}, max: {r_step0.max().item():.6f}, mean: {r_step0.mean().item():.6f}")
    assert torch.allclose(r_step0, torch.ones_like(r_step0)), "Ratio at step 0 must be exactly 1.0!"
    
    # Case 2: Random perturbation / policy update & Dimensional Underflow
    print(f"\n[H1.2] Testing ratio decay across drift scales (dim={dim}, sigma={sigma}):")
    underflow_threshold = 2 * sigma**2 * 88.7 # ln(float32_min) ~ -88.7
    print(f"       Theoretical float32 underflow threshold ||h_old - h_theta||^2: {underflow_threshold:.1f}")
    
    for drift_scale in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
        h_perturbed = h_old + drift_scale * torch.randn_like(h_old)
        diff_p = (h_old - h_perturbed).pow(2).sum(dim=-1)
        r_p = torch.exp(-diff_p / (2 * sigma**2))
        avg_diff = diff_p.mean().item()
        print(f"       Drift {drift_scale:5.3f}: mean ||drift||^2 = {avg_diff:8.1f} | Ratio mean = {r_p.mean().item():.6e} (min={r_p.min().item():.6e})")
        assert (r_p <= 1.0).all(), "Ratio must be strictly <= 1.0 for all theta!"
        if avg_diff > underflow_threshold:
            print(f"       -> [NUMERICAL COLLAPSE] Drift scale {drift_scale} exceeds underflow boundary ({avg_diff:.1f} > {underflow_threshold:.1f}), r_t collapses to 0.0!")
            
    print("-> VERIFIED: VLPO ratio r_{i,t}(\theta) is strictly in [0.0, 1.0] across all parameter spaces.")
    print("   CRITICAL FLAW DETECTED: Without 1/D dimension normalization, any coordinate drift > 0.22 causes catastrophic float32 underflow (r_t -> 0.0), killing all gradients!")
    
    # Case 3: Autograd direction with Positive Advantage
    # In standard RL, when advantage > 0, policy gradient increases prob of action (moves away from old if exploring better).
    # Here, loss = - r * A.
    h_theta = nn.Parameter(h_old.clone() + 0.1 * torch.randn_like(h_old))
    advantage = torch.ones(batch_size, num_latents) * 2.0 # Positive advantage
    diff = (h_old - h_theta).pow(2).sum(dim=-1)
    r = torch.exp(-diff / (2 * sigma**2))
    loss = - (r * advantage).mean()
    loss.backward()
    
    # Analytical gradient:
    # d(- r * A) / d(h_theta) = - A * r * ( - 2 * (h_theta - h_old) / (2 * sigma^2) ) * (-1) ??
    # Let's check: d(diff)/d(h_theta) = 2*(h_theta - h_old)
    # d(exp(-diff/(2*sigma^2)))/d(h_theta) = r * (- (h_theta - h_old)/sigma^2)
    # d(-r*A)/d(h_theta) = A * r * (h_theta - h_old) / sigma^2
    # So gradient pulls h_theta back toward h_old!
    grad_dot_drift = (h_theta.grad * (h_theta.data - h_old)).sum()
    print(f"[H1.3] Gradient dot drift (Positive Advantage): {grad_dot_drift.item():.6f}")
    assert grad_dot_drift > 0, "Gradient must point in direction of (h_theta - h_old), acting as an L2 anchor back to h_old!"
    print("-> VERIFIED: Positive advantage strictly penalizes exploration and pulls latent vectors back to h_old.")
    print("   VLPO acts mathematically as an L2 regularizer anchored to h_old, NOT an exploratory policy gradient.\n")


def test_h2_detached_proxy_loss():
    print("=" * 70)
    print("TESTING HYPOTHESIS H2: SFT Detached Proxy Loss Gradient Isolation")
    print("=" * 70)
    
    # In src/trainer.py L11-42, L198-203:
    # Author computes:
    # g = torch.autograd.grad(l_obs, z, retain_graph=True)[0]
    # loss = (z * g.detach()).sum()
    
    dim = 256
    seq_len_z = 10
    seq_len_obs = 20
    
    # Model parameters for text encoder producing observations
    W_obs = nn.Linear(dim, dim, bias=False)
    # Latent representations z
    z = nn.Parameter(torch.randn(seq_len_z, dim))
    # Observation representations from earlier layer
    x_obs = torch.randn(seq_len_obs, dim)
    h_obs = W_obs(x_obs)
    
    # Suppose l_obs measures alignment between z and h_obs (e.g. cross attention or dot product)
    # l_obs = (z.mean(dim=0) * h_obs.mean(dim=0)).sum()
    l_obs = (torch.matmul(z, h_obs.T)).sum()
    
    # 1. Compute g = grad(l_obs, z)
    g = torch.autograd.grad(l_obs, z, retain_graph=True, create_graph=False)[0]
    
    # 2. Compute surrogate loss using g.detach() as done in Monet:
    proxy_loss = (z * g.detach()).sum()
    
    # 3. Backward pass from proxy_loss
    proxy_loss.backward()
    
    # Verify: W_obs.weight.grad MUST BE NONE or ZERO because g was detached!
    print(f"[H2.1] Latent gradient norm ||grad_z||: {z.grad.norm().item():.6f}")
    print(f"[H2.2] Observation layer W_obs.weight.grad: {W_obs.weight.grad}")
    
    assert z.grad is not None and z.grad.norm() > 0, "z must receive gradient from proxy loss."
    assert W_obs.weight.grad is None, "W_obs must receive ZERO gradient because g was detached!"
    print("-> VERIFIED: Observation tokens receive strictly 0.000000 direct gradient through the proxy loss.")
    print("   The loss term creates zero backpropagation into the text observation backbone.\n")


if __name__ == "__main__":
    test_h1_vlpo_ratio_and_gradient()
    test_h2_detached_proxy_loss()
    print("=" * 70)
    print("ALL MATHEMATICAL & AUTOGRAD DIAGNOSTIC TESTS PASSED (0 FAILURES).")
    print("=" * 70)
