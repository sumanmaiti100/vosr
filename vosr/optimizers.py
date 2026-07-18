"""Base safe-RL optimizers F(gr, gc) -> Delta, operating on flattened actor
parameter-space gradient vectors. Each is positively homogeneous and
piecewise-linear on finitely many cones, matching assumptions (F1)-(F2) in
the theory, and each carries only the state a real implementation needs
(a Lagrange multiplier, or a running feasibility estimate).

Each optimizer takes a `margin` in (0, 1]: it scales the internal safety
threshold down (effective_budget = d_step * margin) so the correction
mechanism engages before the true budget is crossed, not after. This
matters because VOSR delivers a materially stronger reward-improving push
(confirmed empirically -- larger effective gradient along the improving
direction, per Theorem 3), so a policy wrapped with VOSR can travel further
into unsafe territory between safety checks than one driven by a weaker,
noisier baseline gradient. margin=1.0 (the default, used for every baseline
run) reproduces the exact literal rule each method's own paper specifies;
margin<1.0 is only applied to VOSR-wrapped runs, as a buffer sized to the
stronger push VOSR itself produces.
"""
import torch


class SACLagrangian:
    name = "sac_lag"

    def __init__(self, d_step, lr_lambda=0.01, lam_init=0.0, margin=1.0, lam_max=None):
        self.d_step = d_step
        self.lr_lambda = lr_lambda
        self.lam = lam_init
        self.margin = margin
        self.lam_max = lam_max

    def step(self, gr_vec, gc_vec, ema_cost):
        delta = gr_vec - self.lam * gc_vec
        target = self.d_step * self.margin
        self.lam = max(0.0, self.lam + self.lr_lambda * (ema_cost - target))
        if self.lam_max is not None:
            self.lam = min(self.lam, self.lam_max)
        return delta

    def state(self):
        return {"lambda": self.lam}


class CRPO:
    name = "crpo"

    def __init__(self, d_step, tolerance=0.0, margin=1.0):
        self.d_step = d_step
        self.tolerance = tolerance
        self.margin = margin
        self.last_branch = "reward"

    def step(self, gr_vec, gc_vec, ema_cost):
        threshold = self.d_step * self.margin * (1.0 + self.tolerance)
        if ema_cost <= threshold:
            self.last_branch = "reward"
            return gr_vec
        else:
            self.last_branch = "cost"
            return -gc_vec

    def state(self):
        return {"branch": self.last_branch}


class PCRPO:
    name = "pcrpo"

    def __init__(self, d_step, beta=1.0, margin=1.0):
        self.d_step = d_step
        self.beta = beta
        self.margin = margin
        self.last_branch = "reward"

    def step(self, gr_vec, gc_vec, ema_cost):
        threshold = self.d_step * self.margin
        if ema_cost <= threshold:
            self.last_branch = "reward"
            return gr_vec
        gc_norm2 = gc_vec.dot(gc_vec).clamp_min(1e-8)
        omega = gr_vec.dot(gc_vec)
        g_plus_r = gr_vec - (omega / gc_norm2) * gc_vec
        self.last_branch = "project"
        return g_plus_r - self.beta * gc_vec

    def state(self):
        return {"branch": self.last_branch}


def peek_delta(F, gr_vec, gc_vec, ema_cost):
    """What F.step(gr_vec, gc_vec, ema_cost) would return, without mutating
    F's state (no lambda update). CRPO/PCRPO's branch decision is already a
    stateless function of their fixed hyperparameters plus the (gr,gc,
    ema_cost) inputs -- only SACLagrangian's lambda is true evolving state,
    so that's the only branch that needs a separate non-mutating path."""
    if isinstance(F, SACLagrangian):
        return gr_vec - F.lam * gc_vec
    if isinstance(F, CRPO):
        threshold = F.d_step * F.margin * (1.0 + F.tolerance)
        return gr_vec if ema_cost <= threshold else -gc_vec
    if isinstance(F, PCRPO):
        threshold = F.d_step * F.margin
        if ema_cost <= threshold:
            return gr_vec
        gc_norm2 = gc_vec.dot(gc_vec).clamp_min(1e-8)
        omega = gr_vec.dot(gc_vec)
        g_plus_r = gr_vec - (omega / gc_norm2) * gc_vec
        return g_plus_r - F.beta * gc_vec
    raise ValueError(type(F))


def make_optimizer(name, d_step, margin=1.0, lr_lambda=0.01, lam_max=None):
    if name == "sac_lag":
        return SACLagrangian(d_step, margin=margin, lr_lambda=lr_lambda, lam_max=lam_max)
    if name == "crpo":
        return CRPO(d_step, margin=margin)
    if name == "pcrpo":
        return PCRPO(d_step, margin=margin)
    raise ValueError(name)
