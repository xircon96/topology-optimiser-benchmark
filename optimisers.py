"""
optimisers.py
-------------
Pluggable optimisers for density-based topology optimisation, all behind one
interface so that only the update rule differs between them:

    opt = Optimiser(problem, **hyperparams)
    x_new = opt.step(x, dc, dv)

Constraint handling is unified. OC uses its native Lagrangian bisection.
The gradient methods (GD, Adam) operate on an unconstrained latent field z,
mapping to densities via a sigmoid whose bias is bisected each step so that
mean(sigmoid(z + b)) equals the target volume fraction. This is applied
identically to every gradient method, so any difference in outcome is due to
the update rule, not the constraint scheme. The latent field is clamped to a
range where the sigmoid never saturates, preventing silent volume-constraint
failure.

UK English throughout.
"""

from __future__ import annotations
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# Latent clamp: beyond +/-25 the density is within ~1e-11 of 0 or 1, so the
# clamp is physically inert but guarantees the volume-bias bisection can always
# reach the target mean.
Z_CLAMP = 25.0


def _volume_map(z, target_vol):
    """Map latent z to densities meeting the volume constraint via bias bisection."""
    l1, l2 = -50.0, 50.0
    for _ in range(80):
        b = 0.5 * (l1 + l2)
        if _sigmoid(z + b).mean() > target_vol:
            l2 = b
        else:
            l1 = b
    return _sigmoid(z + 0.5 * (l1 + l2))


class OCOptimiser:
    """Classic Optimality Criteria update (Lagrangian multiplier by bisection)."""

    name = "OC"

    def __init__(self, problem, move=0.2, eta=0.5):
        self.p = problem
        self.move = move
        self.eta = eta

    def step(self, x, dc, dv):
        l1, l2 = 1e-9, 1e9
        vol = self.p.volfrac
        xnew = x.copy()
        while (l2 - l1) / (0.5 * (l1 + l2)) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            be = np.maximum(0.0, -dc / (dv * lmid))
            xcand = x * np.power(be, self.eta)
            xnew = np.clip(
                np.clip(xcand, np.maximum(0.0, x - self.move),
                        np.minimum(1.0, x + self.move)),
                0.0, 1.0)
            if xnew.mean() > vol:
                l1 = lmid
            else:
                l2 = lmid
        return xnew


class _LatentGradientBase:
    """Shared latent-field machinery for the gradient methods."""

    def __init__(self, problem):
        self.p = problem
        self.z = None

    def _init_latent(self, x, rng=None):
        xc = np.clip(x, 1e-4, 1 - 1e-4)
        self.z = np.log(xc / (1.0 - xc))
        if rng is not None:
            self.z = self.z + 1e-2 * (rng.random(self.z.shape) - 0.5)

    def _finish(self):
        self.z = np.clip(self.z, -Z_CLAMP, Z_CLAMP)
        return _volume_map(self.z, self.p.volfrac)


class GDOptimiser(_LatentGradientBase):
    """Gradient descent on the latent field (RMS-scaled step)."""

    name = "GD"

    def __init__(self, problem, lr=0.5):
        super().__init__(problem)
        self.lr = lr

    def step(self, x, dc, dv):
        if self.z is None:
            self._init_latent(x, getattr(self, '_rng', None))
        dxdz = x * (1.0 - x)              # chain rule: d(density)/d(latent)
        g = dc * dxdz
        g = g / (np.sqrt(np.mean(g ** 2)) + 1e-12)
        self.z = self.z - self.lr * g
        return self._finish()


class AdamOptimiser(_LatentGradientBase):
    """Adam update on the latent field. Adam's own second-moment estimate
    provides per-element scaling, so the gradient is NOT pre-normalised."""

    name = "Adam"

    def __init__(self, problem, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(problem)
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, x, dc, dv):
        if self.z is None:
            self._init_latent(x, getattr(self, '_rng', None))
            self.m = np.zeros_like(self.z)
            self.v = np.zeros_like(self.z)
        self.t += 1
        dxdz = x * (1.0 - x)
        g = dc * dxdz
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        self.z = self.z - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return self._finish()




class MMAOptimiser:
    """Method of Moving Asymptotes (Svanberg). The field-standard gradient
    method for constrained topology optimisation; handles the volume
    constraint natively rather than via the latent sigmoid map."""

    name = "MMA"

    def __init__(self, problem, move=0.2):
        self.p = problem
        self.move = move
        self.state = None
        self.it = 0

    def step(self, x, dc, dv):
        from mma import MMAState, mma_sub
        n = len(x)
        if self.state is None:
            self.state = MMAState(n)
        gx = x.mean() - self.p.volfrac          # g <= 0 feasible
        dgdx = dv                               # d(mean x)/dx = 1/n per element
        xmin = np.zeros(n)
        xmax = np.ones(n)
        self.it += 1
        xnew = mma_sub(x, dc, gx, dgdx, xmin, xmax, self.state,
                       self.it, move=self.move)
        return np.clip(xnew, 0.0, 1.0)


OPTIMISERS = {
    "OC": OCOptimiser,
    "GD": GDOptimiser,
    "MMA": MMAOptimiser,
    "Adam": AdamOptimiser,  # retained for the optional ML-optimiser study
}
