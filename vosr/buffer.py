import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.c = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.logp_beh = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, c, s2, done, logp_beh):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.c[i] = c
        self.s2[i] = s2; self.done[i] = done; self.logp_beh[i] = logp_beh
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_uniform_idx(self, n):
        return np.random.randint(0, self.size, size=n)

    def to_torch(self, idx):
        return (
            torch.as_tensor(self.s[idx], device=self.device),
            torch.as_tensor(self.a[idx], device=self.device),
            torch.as_tensor(self.r[idx], device=self.device),
            torch.as_tensor(self.c[idx], device=self.device),
            torch.as_tensor(self.s2[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
            torch.as_tensor(self.logp_beh[idx], device=self.device),
        )
