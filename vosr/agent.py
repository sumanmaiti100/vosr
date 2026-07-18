import numpy as np
import torch
import torch.nn.functional as Fnn

from vosr.networks import GaussianPolicy, TwinQ
from vosr.buffer import ReplayBuffer
from vosr.optimizers import make_optimizer
from vosr.scoring import (stamped_actor_gradients, vosr_scores_and_sampler,
                           tilted_softmax, unflatten_like, robust_clip)

VOSR_METHODS = {"vosr"}
BASELINE_SAMPLERS = {"uniform", "safety_per", "uncertainty_per"}


class RunningNorm:
    def __init__(self, dim, eps=1e-4):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps

    def update(self, x):
        batch_mean = x
        self.count += 1
        delta = batch_mean - self.mean
        self.mean += delta / self.count
        self.var += (batch_mean - self.mean) * delta * (self.count - 1) / self.count if self.count > 1 else 0

    def normalize(self, x):
        std = np.sqrt(self.var / max(self.count, 1.0)) + 1e-6
        std = np.maximum(std, 1e-3)
        return (x - self.mean) / std


class Agent:
    def __init__(self, obs_dim, act_dim, base_optimizer, sampler, device,
                 cost_limit=25.0, max_ep_len=1000, gamma=0.99, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, buffer_size=300_000,
                 clip_c=10.0, kappa=0.3, eta=1.0, pool_size=512,
                 ref_size=256, train_bs=256, k_cost_ensemble=3, seed=0,
                 optimizer_margin=1.0, episode_cost_tracking=False,
                 shield_enabled=False, shield_threshold_frac=0.4, shield_k=16,
                 lr_lambda=0.01, lam_max=None):
        self.device = device
        self.gamma, self.tau = gamma, tau
        self.clip_c = clip_c
        self.kappa, self.eta = kappa, eta
        self.pool_size, self.ref_size, self.train_bs = pool_size, ref_size, train_bs
        self.max_ep_len = max_ep_len
        self.d_step = cost_limit / max_ep_len
        self.cost_limit = cost_limit
        self.sampler_name = sampler
        self.rng = np.random.default_rng(seed)
        # Episode-level cost tracking (VOSR only; baselines keep the exact
        # per-step EMA they were already trained and reported with). The
        # constraint Jc <= d is episodic, but a per-step EMA with fast decay
        # crushes sparse/hazard-triggered cost (dense navigation tasks) back
        # to ~0 between rare events, leaving the safety mechanism blind to
        # real episodic violations. Tracking completed-episode totals fixes
        # that without touching how the per-step signal behaves elsewhere.
        self.episode_cost_tracking = episode_cost_tracking
        self.episode_cost_accum = 0.0
        self.ema_episode_cost = 0.0
        F_threshold = cost_limit if episode_cost_tracking else self.d_step
        # Runtime safety shield: gradient-based training correction only
        # shapes average behavior between updates/episodes -- it cannot stop
        # a single bad episode already unfolding, since the stochastic policy
        # can wander into a hazard faster than any training-side signal
        # reacts. Once running episode cost approaches the budget, this
        # switches action selection to best-of-K under the trained cost
        # critic instead of trusting a raw policy sample, capping how much
        # damage one episode can still do.
        self.shield_enabled = shield_enabled
        self.shield_threshold_frac = shield_threshold_frac
        self.shield_k = shield_k

        self.policy = GaussianPolicy(obs_dim, act_dim).to(device)
        self.qr = TwinQ(obs_dim, act_dim, k=2).to(device)
        self.qr_targ = TwinQ(obs_dim, act_dim, k=2).to(device)
        self.qr_targ.load_state_dict(self.qr.state_dict())
        self.qc = TwinQ(obs_dim, act_dim, k=k_cost_ensemble).to(device)
        self.qc_targ = TwinQ(obs_dim, act_dim, k=k_cost_ensemble).to(device)
        self.qc_targ.load_state_dict(self.qc.state_dict())

        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.qr_opt = torch.optim.Adam(self.qr.parameters(), lr=critic_lr)
        self.qc_opt = torch.optim.Adam(self.qc.parameters(), lr=critic_lr)

        self.F = make_optimizer(base_optimizer, F_threshold, margin=optimizer_margin,
                                 lr_lambda=lr_lambda, lam_max=lam_max)
        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, device)
        self.obs_norm = RunningNorm(obs_dim)
        self.ema_cost = 0.0
        self.total_updates = 0
        self.last_wrong_branch = 0.0

    def norm_obs(self, s):
        return self.obs_norm.normalize(s).astype(np.float32)

    @torch.no_grad()
    def act(self, s_np, deterministic=False, running_episode_cost=None):
        s = torch.as_tensor(self.norm_obs(s_np), dtype=torch.float32, device=self.device).unsqueeze(0)
        ep_cost = running_episode_cost if running_episode_cost is not None else self.episode_cost_accum
        if self.shield_enabled and ep_cost >= self.shield_threshold_frac * self.cost_limit:
            # In these environments cost is proximity-based, not action-
            # magnitude-based: standing still inside a hazard zone still
            # accrues cost every step. The shield must find an action that
            # actively moves away, not just stop. Half the candidates come
            # from the policy (informed, but possibly hazard-biased itself);
            # half are pure-random over the action space, for direction
            # diversity the policy's own distribution may lack. Rank all of
            # them by the trained cost critic and take the best.
            k_policy = self.shield_k // 2
            k_random = self.shield_k - k_policy
            s_rep = s.repeat(k_policy, 1)
            a_policy, _, _ = self.policy.sample(s_rep)
            a_random = (torch.rand(k_random, self.policy.act_dim, device=self.device) * 2 - 1)
            a_cands = torch.cat([a_policy, a_random], dim=0)
            s_rep_all = s.repeat(self.shield_k, 1)
            qc_pred = self.qc.forward_mean(s_rep_all, a_cands)
            best_idx = torch.argmin(qc_pred)
            a = a_cands[best_idx:best_idx + 1]
            logp = self.policy.logp_of_action(s, a)
        elif deterministic:
            mu, _ = self.policy.forward(s)
            a = torch.tanh(mu)
            logp = self.policy.logp_of_action(s, a)
        else:
            a, logp, _ = self.policy.sample(s)
        return a.squeeze(0).cpu().numpy(), float(logp.item())

    def observe(self, s, a, r, c, s2, done, logp_beh, episode_done=False):
        self.obs_norm.update(s)
        self.ema_cost = 0.99 * self.ema_cost + 0.01 * c
        if self.episode_cost_tracking:
            self.episode_cost_accum += c
            if episode_done:
                self.ema_episode_cost = 0.9 * self.ema_episode_cost + 0.1 * self.episode_cost_accum
                self.episode_cost_accum = 0.0
        self.buffer.add(self.norm_obs(s), a, r, c, self.norm_obs(s2), float(done), logp_beh)

    @property
    def feasibility_signal(self):
        return self.ema_episode_cost if self.episode_cost_tracking else self.ema_cost

    # ---------------- critic / advantage helpers ----------------
    def _fresh_values(self, s):
        with torch.no_grad():
            a_fresh, _, _ = self.policy.sample(s)
            vr = self.qr.forward_min(s, a_fresh)
            vc = self.qc.forward_mean(s, a_fresh)
        return vr, vc

    def _importance_weight(self, s, a, logp_beh):
        with torch.no_grad():
            logp_cur = self.policy.logp_of_action(s, a)
            w = torch.exp((logp_cur - logp_beh).clamp(-10, 10))
            w = w.clamp(1.0 / self.clip_c, self.clip_c)
        return w

    def _update_critics(self, s, a, r, c, s2, done, stamp):
        with torch.no_grad():
            a2, _, _ = self.policy.sample(s2)
            yr = r + self.gamma * (1 - done) * self.qr_targ.forward_min(s2, a2)
            yc = c + self.gamma * (1 - done) * self.qc_targ.forward_mean(s2, a2)
        stamp_n = (stamp / stamp.mean().clamp_min(1e-6)).detach()

        qr_preds = self.qr.forward_all(s, a)
        loss_r = (stamp_n.unsqueeze(0) * (qr_preds - yr.unsqueeze(0)).pow(2)).mean()
        self.qr_opt.zero_grad(set_to_none=True)
        loss_r.backward()
        self.qr_opt.step()

        qc_preds = self.qc.forward_all(s, a)
        mask = (torch.rand_like(qc_preds) < 0.85).float()
        loss_c = (mask * stamp_n.unsqueeze(0) * (qc_preds - yc.unsqueeze(0)).pow(2)).sum() / mask.sum().clamp_min(1.0)
        self.qc_opt.zero_grad(set_to_none=True)
        loss_c.backward()
        self.qc_opt.step()

        for net, targ in ((self.qr, self.qr_targ), (self.qc, self.qc_targ)):
            with torch.no_grad():
                for p, pt in zip(net.parameters(), targ.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss_r.item()), float(loss_c.item())

    def _advantages(self, s, a):
        vr, vc = self._fresh_values(s)
        qr_v = self.qr.forward_min(s, a)
        qc_v = self.qc.forward_mean(s, a)
        return (qr_v - vr).detach(), (qc_v - vc).detach()

    # ---------------- training step ----------------
    def train_step(self):
        if self.buffer.size < max(self.ref_size, self.train_bs) + 10:
            return None
        if self.sampler_name in VOSR_METHODS:
            return self._train_step_vosr()
        return self._train_step_baseline()

    def _actor_apply(self, delta_vec):
        params = dict(self.policy.named_parameters())
        delta_dict = unflatten_like(delta_vec, params)
        self.actor_opt.zero_grad(set_to_none=True)
        for name, p in self.policy.named_parameters():
            p.grad = -delta_dict[name].clone()
        self.actor_opt.step()

    def _train_step_baseline(self):
        pool_n = min(self.pool_size, self.buffer.size)
        idx_pool = self.buffer.sample_uniform_idx(pool_n)
        s, a, r, c, s2, done, logp_beh = self.buffer.to_torch(idx_pool)
        w = self._importance_weight(s, a, logp_beh)

        if self.sampler_name == "uniform":
            probs = torch.full((pool_n,), 1.0 / pool_n, device=self.device)
        else:
            with torch.no_grad():
                a2, _, _ = self.policy.sample(s2)
                if self.sampler_name == "td_per":
                    # classic PER (Schaul et al. 2015): reward-critic TD error
                    td = r + self.gamma * (1 - done) * self.qr_targ.forward_min(s2, a2) - self.qr.forward_min(s, a)
                    priority = td.abs() + 1e-3
                elif self.sampler_name == "safety_per":
                    td = c + self.gamma * (1 - done) * self.qc_targ.forward_mean(s2, a2) - self.qc.forward_mean(s, a)
                    priority = td.abs() + 1e-3
                else:  # uncertainty_per
                    priority = self.qc.forward_std(s, a) + 1e-3
                p_alpha = priority.pow(0.6)
                probs = p_alpha / p_alpha.sum()

        train_n = min(self.train_bs, pool_n)
        sel = torch.multinomial(probs, train_n, replacement=True)
        b_i = 1.0 / pool_n
        stamp = (b_i / probs[sel].clamp_min(1e-8)) * w[sel]

        s_m, a_m, r_m, c_m, s2_m, done_m = s[sel], a[sel], r[sel], c[sel], s2[sel], done[sel]
        loss_r, loss_c = self._update_critics(s_m, a_m, r_m, c_m, s2_m, done_m, stamp)

        adv_r, adv_c = self._advantages(s_m, a_m)
        params = {k: v.detach().clone() for k, v in self.policy.named_parameters()}
        gr_vec, gc_vec = stamped_actor_gradients(self.policy, params, s_m, a_m, adv_r, adv_c, stamp)
        delta = self.F.step(gr_vec, gc_vec, self.feasibility_signal)
        self._actor_apply(delta)
        self.total_updates += 1
        return {"loss_r": loss_r, "loss_c": loss_c, "lambda": getattr(self.F, "lam", None),
                "branch": getattr(self.F, "last_branch", None), "ema_cost": self.ema_cost}

    def _train_step_vosr(self):
        # Stage 1: reference batch -> gr, gc direction estimate (uniform sampling, b=q)
        ref_n = min(self.ref_size, self.buffer.size)
        idx_ref = self.buffer.sample_uniform_idx(ref_n)
        s_r, a_r, r_r, c_r, s2_r, done_r, logp_r = self.buffer.to_torch(idx_ref)
        w_r = self._importance_weight(s_r, a_r, logp_r)
        adv_r_r, adv_c_r = self._advantages(s_r, a_r)
        params = {k: v.detach().clone() for k, v in self.policy.named_parameters()}
        gr_vec, gc_vec = stamped_actor_gradients(self.policy, params, s_r, a_r, adv_r_r, adv_c_r, w_r)

        # Stage 2: score a fresh pool via JVP projections, build tilted sampler
        pool_n = min(self.pool_size, self.buffer.size)
        idx_pool = self.buffer.sample_uniform_idx(pool_n)
        s_p, a_p, r_p, c_p, s2_p, done_p, logp_p = self.buffer.to_torch(idx_pool)
        w_p = self._importance_weight(s_p, a_p, logp_p)
        adv_r_p, adv_c_p = self._advantages(s_p, a_p)

        proj_c, proj_r, gc_norm, gr_plus_norm = vosr_scores_and_sampler(
            self.policy, params, gr_vec, gc_vec, s_p, a_p, w_p, self.kappa, self.eta)
        s_c = robust_clip(w_p * adv_c_p * proj_c)
        s_r_score = robust_clip(w_p * adv_r_p * proj_r)
        rho = torch.sqrt(s_c.pow(2) + self.kappa * s_r_score.pow(2) + 1e-12)
        q_star = tilted_softmax(rho, self.eta)
        ess = float(1.0 / (q_star.pow(2).sum().item() + 1e-12))
        self.last_ess_frac = ess / pool_n

        train_n = min(self.train_bs, pool_n)
        sel = torch.multinomial(q_star, train_n, replacement=True)
        b_i = 1.0 / pool_n
        stamp = (b_i / q_star[sel].clamp_min(1e-8)) * w_p[sel]

        s_m, a_m, r_m, c_m, s2_m, done_m = s_p[sel], a_p[sel], r_p[sel], c_p[sel], s2_p[sel], done_p[sel]
        loss_r, loss_c = self._update_critics(s_m, a_m, r_m, c_m, s2_m, done_m, stamp)

        # Stage 3: re-estimate gr, gc from the stamped, sharpened minibatch
        adv_r_m, adv_c_m = self._advantages(s_m, a_m)
        gr_vec2, gc_vec2 = stamped_actor_gradients(self.policy, params, s_m, a_m, adv_r_m, adv_c_m, stamp)
        delta = self.F.step(gr_vec2, gc_vec2, self.feasibility_signal)
        hard_override = False
        if self.episode_cost_tracking and self.feasibility_signal > 2.0 * self.cost_limit:
            # Circuit breaker: gradient blending alone can lag behind a severe
            # episodic overshoot (F's own state, e.g. SAC-Lag's lambda, still
            # updates normally above -- this only overrides the applied step).
            delta = -gc_vec2
            hard_override = True
        self._actor_apply(delta)
        self.total_updates += 1
        return {"loss_r": loss_r, "loss_c": loss_c, "lambda": getattr(self.F, "lam", None),
                "branch": "hard_override" if hard_override else getattr(self.F, "last_branch", None),
                "ema_cost": self.ema_cost, "ema_episode_cost": self.ema_episode_cost,
                "gc_norm": float(gc_norm.item()), "gr_plus_norm": float(gr_plus_norm.item()),
                "sigma_c2": float(torch.var(s_c).item()), "ess_frac": self.last_ess_frac}
