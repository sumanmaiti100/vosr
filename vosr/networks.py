"""Actor and critic networks. The actor exposes a purely-functional log-prob
so that torch.func.jvp/vmap can compute directional derivatives ⟨grad log pi, v⟩
cheaply (forward-mode AD) for the VOSR scoring rule, without materializing a
full per-sample parameter-space gradient.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

LOG_STD_MIN, LOG_STD_MAX = -2.0, 2.0
EPS = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.l0 = nn.Linear(obs_dim, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, s):
        h = torch.tanh(self.l0(s))
        h = torch.tanh(self.l1(h))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        pre = mu + std * torch.randn_like(mu)
        a = torch.tanh(pre)
        logp = gaussian_logp(pre, mu, log_std) - torch.log(1 - a.pow(2) + EPS).sum(-1)
        return a, logp, pre

    def logp_of_action(self, s, a_env):
        """log pi(a_env | s) for an action already in env (tanh-squashed) space."""
        pre = torch.atanh(a_env.clamp(-1 + EPS, 1 - EPS))
        mu, log_std = self.forward(s)
        logp = gaussian_logp(pre, mu, log_std) - torch.log(1 - a_env.pow(2) + EPS).sum(-1)
        return logp

    # ---- functional form for jvp/vmap ----
    def params_dict(self):
        return {k: v.detach().clone() for k, v in self.named_parameters()}

    def functional_logp(self, params, s_i, a_env_i):
        """Pure function of params for a SINGLE (unbatched) transition."""
        h = torch.tanh(Fnn.linear(s_i, params['l0.weight'], params['l0.bias']))
        h = torch.tanh(Fnn.linear(h, params['l1.weight'], params['l1.bias']))
        mu = Fnn.linear(h, params['mu.weight'], params['mu.bias'])
        log_std = Fnn.linear(h, params['log_std.weight'], params['log_std.bias']).clamp(LOG_STD_MIN, LOG_STD_MAX)
        pre = torch.atanh(a_env_i.clamp(-1 + EPS, 1 - EPS))
        logp = gaussian_logp(pre, mu, log_std) - torch.log(1 - a_env_i.pow(2) + EPS).sum(-1)
        return logp


def gaussian_logp(x, mu, log_std):
    std = log_std.exp()
    var = std * std
    logp = -0.5 * (((x - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
    return logp.sum(-1)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.l0 = nn.Linear(obs_dim + act_dim, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = Fnn.relu(self.l0(x))
        h = Fnn.relu(self.l1(h))
        return self.l2(h).squeeze(-1)


class TwinQ(nn.Module):
    """K-member critic ensemble (K=2 gives the usual twin-Q trick; K>2 also
    supplies an uncertainty signal for the Uncertainty-PER baseline)."""
    def __init__(self, obs_dim, act_dim, k=2, hidden=128):
        super().__init__()
        self.qs = nn.ModuleList([QNetwork(obs_dim, act_dim, hidden) for _ in range(k)])
        self.k = k

    def forward_all(self, s, a):
        return torch.stack([q(s, a) for q in self.qs], dim=0)  # [K, B]

    def forward_min(self, s, a):
        return self.forward_all(s, a).min(dim=0).values

    def forward_mean(self, s, a):
        return self.forward_all(s, a).mean(dim=0)

    def forward_std(self, s, a):
        return self.forward_all(s, a).std(dim=0)
