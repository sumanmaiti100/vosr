"""VOSR core math: per-transition scores s^c_i, s^r_i via forward-mode JVP,
the closed-form variance-optimal tilted sampler q*_eps, and the flattened
parameter-space gradient estimators gr, gc (Lemma 1's X_i, Y_i, batch-averaged).
"""
import torch
from torch.func import jvp, vmap


def flatten(d):
    return torch.cat([v.reshape(-1) for v in d.values()])


def unflatten_like(vec, ref_dict):
    out, i = {}, 0
    for k, v in ref_dict.items():
        n = v.numel()
        out[k] = vec[i:i + n].view_as(v)
        i += n
    return out


def stamped_actor_gradients(policy, params, s, a, adv_r, adv_c, stamp):
    """Batch-mean stamped score-function gradient vectors gr_hat, gc_hat
    (flattened over actor params), matching X_i = (b/q) w_i u_i averaged,
    with u_i = Q_r(s_i,a_i) * grad_theta log pi(a_i|s_i) and adv used as a
    variance-reducing (unbiased) baseline-corrected version of Q.
    stamp already folds in b(i)/q(i) * w_i.
    """
    params_leaf = {k: v.detach().requires_grad_(True) for k, v in params.items()}
    logp = policy_logp_batch(policy, params_leaf, s, a)  # [B]
    surrogate_r = (stamp * adv_r.detach() * logp).mean()
    surrogate_c = (stamp * adv_c.detach() * logp).mean()
    plist = list(params_leaf.values())
    gr = torch.autograd.grad(surrogate_r, plist, retain_graph=True)
    gc = torch.autograd.grad(surrogate_c, plist)
    gr_vec = torch.cat([g.reshape(-1) for g in gr])
    gc_vec = torch.cat([g.reshape(-1) for g in gc])
    return gr_vec.detach(), gc_vec.detach()


def policy_logp_batch(policy, params, s, a):
    mu_h = torch.tanh(torch.nn.functional.linear(s, params['l0.weight'], params['l0.bias']))
    mu_h = torch.tanh(torch.nn.functional.linear(mu_h, params['l1.weight'], params['l1.bias']))
    mu = torch.nn.functional.linear(mu_h, params['mu.weight'], params['mu.bias'])
    from vosr.networks import gaussian_logp, EPS, LOG_STD_MIN, LOG_STD_MAX
    log_std = torch.nn.functional.linear(mu_h, params['log_std.weight'], params['log_std.bias']).clamp(LOG_STD_MIN, LOG_STD_MAX)
    pre = torch.atanh(a.clamp(-1 + EPS, 1 - EPS))
    logp = gaussian_logp(pre, mu, log_std) - torch.log(1 - a.pow(2) + EPS).sum(-1)
    return logp


def directional_score(policy, params, direction_dict, s, a):
    """Vectorized ⟨grad_theta log pi(a_i|s_i), direction⟩ for every i via
    forward-mode JVP, batched with vmap. O(pool) forward passes, no backward."""
    def per_sample(s_i, a_i):
        f = lambda p: policy.functional_logp(p, s_i, a_i)
        _, out = jvp(f, (params,), (direction_dict,))
        return out
    return vmap(per_sample, in_dims=(0, 0))(s, a)


def conflict_free_reward_grad(gr_vec, gc_vec, eps=1e-8):
    gc_norm2 = gc_vec.dot(gc_vec).clamp_min(eps)
    omega = gr_vec.dot(gc_vec)
    g_plus_r = gr_vec - (omega / gc_norm2) * gc_vec
    return g_plus_r


def unit(v, eps=1e-8):
    n = v.norm().clamp_min(eps)
    return v / n, n


def vosr_scores_and_sampler(policy, params, gr_vec, gc_vec, s_pool, a_pool, w_pool, kappa, eta):
    """Returns ρ, q*_eps (categorical probs over the pool), and diagnostics."""
    gc_hat, gc_norm = unit(gc_vec)
    g_plus_r = conflict_free_reward_grad(gr_vec, gc_vec)
    gr_hat, gr_plus_norm = unit(g_plus_r)

    gc_dir = unflatten_like(gc_hat, params)
    gr_dir = unflatten_like(gr_hat, params)

    proj_c = directional_score(policy, params, gc_dir, s_pool, a_pool)  # <grad logp, gc_hat>
    proj_r = directional_score(policy, params, gr_dir, s_pool, a_pool)  # <grad logp, gr+_hat>
    return proj_c, proj_r, gc_norm, gr_plus_norm


def robust_clip(x, k=6.0):
    """Winsorize to +/- k robust-std-devs (via MAD) of the median. A safety
    net against heavy-tailed outliers in the score (e.g. from stale replay
    actions whose score-function gradient blows up as the policy's std
    shrinks) that plain standardization can't fix, since standardizing
    changes scale but not the shape of a heavy-tailed distribution.
    """
    med = x.median()
    mad = (x - med).abs().median().clamp_min(1e-6)
    robust_sd = mad * 1.4826
    return x.clamp(med - k * robust_sd, med + k * robust_sd)


def tilted_softmax(rho, eta):
    """Softmax(rho/eta) per Prop. 3, but rho is first standardized to zero
    mean / unit std within the pool. Raw rho = |Q*<grad logp, dir>| has scale
    that swings by orders of magnitude across environments and training
    stages (unbounded Q-values x unbounded score-function projections), so a
    fixed eta on raw rho makes the softmax numerically collapse onto one or
    two transitions -- exactly the degenerate-sampling failure Prop. 3's
    trust region is meant to prevent. Standardizing makes eta a scale-free
    "how many std-devs of score difference before favoring a transition"
    knob, restoring the intended smooth interpolation between b (eta large)
    and the greedy sqrt(A_i)-proportional optimum (eta small).
    """
    mu = rho.mean()
    sd = rho.std().clamp_min(1e-6)
    z = (rho - mu) / sd
    logits = z / max(eta, 1e-6)
    logits = logits - logits.max()
    p = torch.softmax(logits, dim=0)
    return p
