"""Direct empirical test of Theorem 3 / Corollary 1: for a shared, low-noise
"ground truth" gradient pair (from a large reference batch) and a shared
policy/critic snapshot, draw many small minibatches under each of the 5
candidate samplers, apply the SAME base optimizer F to each (without
mutating F's real training state), and measure the effective-gradient
alignment <Delta*, mean(Delta_q)> / ||Delta*||^2 -- how much of the true,
low-noise update each sampler's noisy minibatches actually deliver on
average. This is the metric Corollary 1 bounds; unlike sigma_c^2 alone, it
also reflects how the *base optimizer's* branch decision reacts to sampler
noise, which is the part of the causal chain sigma_c^2 alone doesn't show.
"""
import csv
import os
import sys

import numpy as np
import safety_gymnasium as sg

from vosr.agent import Agent
from vosr.scoring import (stamped_actor_gradients, directional_score, conflict_free_reward_grad,
                           unit, unflatten_like, tilted_softmax, robust_clip)
from vosr.optimizers import peek_delta
from vosr.train import ENV_COST_LIMIT, ENV_MAX_EP_LEN
import torch

SEEDS = [0, 1, 2]
TRAIN_STEPS = 15000
CHECKPOINT_EVERY = 3000
START_STEPS = 1000
POOL_N = 1024
REF_N = 4096
M_DRAWS = 60
LOG_DIR = "alignment_diagnostic_logs"


def priority_probs(agent, s, a, r, c, s2, done, kind):
    with torch.no_grad():
        a2, _, _ = agent.policy.sample(s2)
        if kind == "td_per":
            td = r + agent.gamma * (1 - done) * agent.qr_targ.forward_min(s2, a2) - agent.qr.forward_min(s, a)
            priority = td.abs() + 1e-3
        elif kind == "safety_per":
            td = c + agent.gamma * (1 - done) * agent.qc_targ.forward_mean(s2, a2) - agent.qc.forward_mean(s, a)
            priority = td.abs() + 1e-3
        else:
            priority = agent.qc.forward_std(s, a) + 1e-3
        p_alpha = priority.pow(0.6)
        return p_alpha / p_alpha.sum()


def measure_alignment(agent, pool_n=POOL_N, ref_n=REF_N, m_draws=M_DRAWS):
    if agent.buffer.size < max(pool_n, ref_n) + 10:
        return None

    idx_ref = agent.buffer.sample_uniform_idx(min(ref_n, agent.buffer.size))
    s_ref, a_ref, r_ref, c_ref, s2_ref, done_ref, logp_ref = agent.buffer.to_torch(idx_ref)
    w_ref = agent._importance_weight(s_ref, a_ref, logp_ref)
    adv_r_ref, adv_c_ref = agent._advantages(s_ref, a_ref)
    params = {k: v.detach().clone() for k, v in agent.policy.named_parameters()}
    gr_true, gc_true = stamped_actor_gradients(agent.policy, params, s_ref, a_ref, adv_r_ref, adv_c_ref, w_ref)

    feas = agent.feasibility_signal
    delta_star = peek_delta(agent.F, gr_true, gc_true, feas)
    star_norm2 = delta_star.dot(delta_star).item()
    if star_norm2 < 1e-10:
        return None

    idx_pool = agent.buffer.sample_uniform_idx(pool_n)
    s_p, a_p, r_p, c_p, s2_p, done_p, logp_p = agent.buffer.to_torch(idx_pool)
    w_p = agent._importance_weight(s_p, a_p, logp_p)
    adv_r_p, adv_c_p = agent._advantages(s_p, a_p)

    gc_hat, _ = unit(gc_true)
    g_plus_r = conflict_free_reward_grad(gr_true, gc_true)
    gr_hat, _ = unit(g_plus_r)
    gc_dir = unflatten_like(gc_hat, params)
    gr_dir = unflatten_like(gr_hat, params)
    proj_c = directional_score(agent.policy, params, gc_dir, s_p, a_p)
    proj_r = directional_score(agent.policy, params, gr_dir, s_p, a_p)
    s_c = robust_clip(w_p * adv_c_p * proj_c)
    s_r = robust_clip(w_p * adv_r_p * proj_r)
    rho = torch.sqrt(s_c.pow(2) + agent.kappa * s_r.pow(2) + 1e-12)
    vosr_q = tilted_softmax(rho, agent.eta)
    uniform_q = torch.full((pool_n,), 1.0 / pool_n, device=s_p.device)

    results = {}
    for kind in ["uniform", "td_per", "safety_per", "uncertainty_per", "vosr"]:
        if kind == "uniform":
            q = uniform_q
        elif kind == "vosr":
            q = vosr_q
        else:
            q = priority_probs(agent, s_p, a_p, r_p, c_p, s2_p, done_p, kind)

        deltas = []
        for _ in range(m_draws):
            sel = torch.multinomial(q, agent.train_bs, replacement=True)
            b_i = 1.0 / pool_n
            stamp = (b_i / q[sel].clamp_min(1e-8)) * w_p[sel]
            gr_d, gc_d = stamped_actor_gradients(agent.policy, params, s_p[sel], a_p[sel],
                                                  adv_r_p[sel], adv_c_p[sel], stamp)
            deltas.append(peek_delta(agent.F, gr_d, gc_d, feas))
        delta_bar = torch.stack(deltas).mean(0)
        alignment = (delta_star @ delta_bar).item() / star_norm2
        results[kind] = alignment
    return results


def run_one(env_id, seed, base_optimizer="sac_lag"):
    device = "cpu"
    env = sg.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = Agent(obs_dim, act_dim, base_optimizer, "vosr", device,
                  cost_limit=ENV_COST_LIMIT[env_id], max_ep_len=ENV_MAX_EP_LEN[env_id], seed=seed)

    os.makedirs(LOG_DIR, exist_ok=True)
    run_name = f"{env_id}__{base_optimizer}__seed{seed}"
    f = open(os.path.join(LOG_DIR, run_name + ".csv"), "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "uniform", "td_per", "safety_per", "uncertainty_per", "vosr"])

    s, info = env.reset(seed=seed)
    ep_len = 0
    step = 0
    while step < TRAIN_STEPS:
        if step < START_STEPS:
            a = env.action_space.sample()
            s_t = torch.as_tensor(agent.norm_obs(s), dtype=torch.float32).unsqueeze(0)
            logp = float(agent.policy.logp_of_action(s_t, torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)).item())
        else:
            a, logp = agent.act(s)
        s2, r, c, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        agent.observe(s, a, r, c, s2, float(terminated), logp)
        ep_len += 1
        s = s2
        step += 1
        if done or ep_len >= ENV_MAX_EP_LEN[env_id]:
            s, info = env.reset()
            ep_len = 0
        if step >= START_STEPS and step % 4 == 0:
            agent.train_step()
        if step % CHECKPOINT_EVERY == 0:
            res = measure_alignment(agent)
            if res is not None:
                writer.writerow([step, res["uniform"], res["td_per"], res["safety_per"],
                                  res["uncertainty_per"], res["vosr"]])
                f.flush()
    env.close()
    f.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--optimizer", default="sac_lag", choices=["sac_lag", "crpo", "pcrpo"])
    args = p.parse_args()
    run_one(args.env, args.seed, args.optimizer)
