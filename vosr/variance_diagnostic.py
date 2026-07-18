"""Direct empirical test of Theorem 1: for a shared pool of transitions and a
shared (semi-trained) critic/policy snapshot, compute the EXACT closed-form
variance sigma_c^2(q) for five candidate samplers -- uniform, TD-PER,
Safety-PER, Uncertainty-PER, and VOSR's tilted sampler -- via the formula

    sigma_c^2(q) = sum_i b(i)^2 (s^c_i)^2 / q(i)  -  ||g_c||^2

(Theorem 1), not by noisy Monte-Carlo resampling. Since b(i)=1/N and the
scores s^c_i are fixed once the pool is drawn, this is a deterministic
number for each q -- the cleanest possible test of "does VOSR minimize
gradient variance", isolated from every downstream RL confound (optimizer
choice, safety tuning, training seed noise) that muddies reward/cost tables.

This trains lightweight, throwaway agents purely to get realistic (not
random-init) critics -- it does not touch any existing run data or figures.
"""
import csv
import os
import sys
import time

import numpy as np
import safety_gymnasium as sg

from vosr.agent import Agent
from vosr.scoring import (stamped_actor_gradients, directional_score, conflict_free_reward_grad,
                           unit, unflatten_like, tilted_softmax, robust_clip)
from vosr.train import ENV_COST_LIMIT, ENV_MAX_EP_LEN
from vosr.full_campaign import TARGET_STEPS, EVAL_INTERVAL
import torch

ENVS = ["SafetyPointGoal1-v0", "SafetyPointCircle1-v0", "SafetyPointButton1-v0", "SafetyPointPush1-v0",
        "SafetyAntVelocity-v1", "SafetyHalfCheetahVelocity-v1", "SafetyHopperVelocity-v1",
        "SafetyWalker2dVelocity-v1", "SafetyHumanoidVelocity-v1", "SafetySwimmerVelocity-v1"]
SEEDS = [0, 1, 2]
START_STEPS = 1000
POOL_N = 1024
LOG_DIR = "variance_diagnostic_logs_full"


def priority_probs(agent, s, a, r, c, s2, done, kind):
    with torch.no_grad():
        a2, _, _ = agent.policy.sample(s2)
        if kind == "td_per":
            td = r + agent.gamma * (1 - done) * agent.qr_targ.forward_min(s2, a2) - agent.qr.forward_min(s, a)
            priority = td.abs() + 1e-3
        elif kind == "safety_per":
            td = c + agent.gamma * (1 - done) * agent.qc_targ.forward_mean(s2, a2) - agent.qc.forward_mean(s, a)
            priority = td.abs() + 1e-3
        else:  # uncertainty_per
            priority = agent.qc.forward_std(s, a) + 1e-3
        p_alpha = priority.pow(0.6)
        return p_alpha / p_alpha.sum()


def measure_variances(agent, pool_n=POOL_N):
    if agent.buffer.size < pool_n + 10:
        return None
    idx = agent.buffer.sample_uniform_idx(pool_n)
    s, a, r, c, s2, done, logp_beh = agent.buffer.to_torch(idx)
    w = agent._importance_weight(s, a, logp_beh)
    adv_r, adv_c = agent._advantages(s, a)
    params = {k: v.detach().clone() for k, v in agent.policy.named_parameters()}
    gr_vec, gc_vec = stamped_actor_gradients(agent.policy, params, s, a, adv_r, adv_c, w)

    gc_hat, gc_norm = unit(gc_vec)
    g_plus_r = conflict_free_reward_grad(gr_vec, gc_vec)
    gr_hat, gr_plus_norm = unit(g_plus_r)
    gc_dir = unflatten_like(gc_hat, params)
    gr_dir = unflatten_like(gr_hat, params)
    proj_c = directional_score(agent.policy, params, gc_dir, s, a)
    proj_r = directional_score(agent.policy, params, gr_dir, s, a)

    s_c = robust_clip(w * adv_c * proj_c).double()
    s_r = robust_clip(w * adv_r * proj_r).double()
    # kappa=0 here deliberately: Theorem 1 with kappa>0 minimizes the BLENDED
    # objective sigma_c^2 + kappa*sigma_r^2, not sigma_c^2 alone -- so the
    # agent's practical, reward-aware kappa (0.15-0.7) is the wrong thing to
    # test against a pure sigma_c^2 claim. kappa=0 is the "pure safety law"
    # (Theorem 2 / the degenerate-geometry remark), the exact case that
    # provably minimizes cost-gradient variance alone.
    rho_pure = s_c.abs()

    N = pool_n
    b_i = 1.0 / N
    gc_est = s_c.mean()  # ||g_c|| estimate under uniform b (Prop. 1)
    gc_est2 = (gc_est ** 2).item()

    def sigma_c2(q_probs):
        # where q(i)=0, s^c_i is also 0 there (Theorem 1 excludes such points
        # from q*'s support) -- the ratio's true limit is 0, not the 0/0 NaN
        # a literal division produces, so clamp the denominator instead of
        # masking: numerator is 0 there too, so a tiny floor forces the
        # correct zero contribution without perturbing any nonzero term.
        val = ((b_i ** 2) * s_c.pow(2) / q_probs.clamp_min(1e-300)).sum().item() - gc_est2
        return val

    results = {}
    uniform_q = torch.full((N,), 1.0 / N, dtype=torch.float64, device=s.device)
    results["uniform"] = sigma_c2(uniform_q)
    for kind in ["td_per", "safety_per", "uncertainty_per"]:
        q = priority_probs(agent, s, a, r, c, s2, done, kind).double()
        results[kind] = sigma_c2(q)
    vosr_q = tilted_softmax(rho_pure, agent.eta).double()
    results["vosr"] = sigma_c2(vosr_q)
    # untempered Theorem-1 (kappa=0) optimum: q*(i) proportional to |s^c_i|
    opt_q = rho_pure / rho_pure.sum()
    results["vosr_theoretical_optimum"] = sigma_c2(opt_q)
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

    train_steps = TARGET_STEPS[env_id]
    checkpoint_every = EVAL_INTERVAL[env_id]

    os.makedirs(LOG_DIR, exist_ok=True)
    run_name = f"{env_id}__{base_optimizer}__seed{seed}"
    f = open(os.path.join(LOG_DIR, run_name + ".csv"), "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "uniform", "td_per", "safety_per", "uncertainty_per", "vosr", "vosr_theoretical_optimum"])

    s, info = env.reset(seed=seed)
    ep_len = 0
    step = 0
    while step < train_steps:
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
        if step % checkpoint_every == 0:
            res = measure_variances(agent)
            if res is not None:
                writer.writerow([step, res["uniform"], res["td_per"], res["safety_per"],
                                  res["uncertainty_per"], res["vosr"], res["vosr_theoretical_optimum"]])
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
