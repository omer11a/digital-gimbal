import torch
import numpy as np

EPS = 1e-8

class GenGS(torch.nn.Module):
    DEFAULT_N = 1500
    DEFAULT_TAU = 1

    def __init__(self, n=DEFAULT_N):
        super().__init__()

        k = torch.arange(n, requires_grad=False, dtype=torch.float)
        self.register_buffer('_k', k)
        self.register_buffer('_lgamma', torch.lgamma(k[:-1] + 1))

    def forward(self, x, tau=DEFAULT_TAU):
        x = x.unsqueeze(-1)
        logits = torch.log(x + EPS) * self._k[:-1] - x - self._lgamma
        pi = torch.exp(logits)
        pi_remainder = torch.nn.functional.relu(1 - pi.sum(-1, keepdim=True))
        logit_remainder = torch.log(pi_remainder + EPS)
        logits = torch.cat((logits, logit_remainder, ), dim=-1)
        w = torch.nn.functional.gumbel_softmax(logits, tau)
        return w.matmul(self._k)

class ReparameterizedPoisson(torch.nn.Module):
    DEFAULT_THRESHOLD = 1000
    DEFAULT_MIN_TAU = 0.1
    DEFAULT_R = 1e-5

    def __init__(
        self,
        n=GenGS.DEFAULT_N,
        threshold=DEFAULT_THRESHOLD,
        min_tau=DEFAULT_MIN_TAU,
        r=DEFAULT_R
    ):
        super().__init__()
        self._gen_gs = GenGS(n=n)
        self._threshold = threshold
        self._min_tau = min_tau
        self._r = r

    def forward(self, x, t=None):
        x = torch.nn.functional.relu(x)
        if (t is None) or (not self.training):
            return torch.poisson(x)

        tau = max(self._min_tau, np.exp(-self._r * t))
        gen_gs = self._gen_gs(x, tau=tau)
        normal = x + torch.normal(0, 1, x.shape, device=x.device) * torch.sqrt(x + EPS)
        return torch.where(x <= self._threshold, gen_gs, normal)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad

round_ste = RoundSTE.apply
