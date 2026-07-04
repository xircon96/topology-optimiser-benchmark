"""
================================================================================
 OPTIMISER-INDUCED NON-UNIQUENESS IN DENSITY-BASED TOPOLOGY OPTIMISATION
 A single self-contained, reproducible benchmark study (SIMP, 2D compliance).
================================================================================

 One file containing:
   - SIMP physics (FE solve, sensitivities, density filter)
   - Three optimisers behind one interface: OC, GD, MMA
   - A self-contained Method of Moving Asymptotes
   - A fixed-horizon driver with post-hoc convergence detection
   - Topology-difference metrics
   - A seed-sweep experiment harness
   - A full visualisation suite (per-iteration curves, distributions,
     agreement matrices, design grids)

 Run:  python topopt_study.py
 All charts and CSV/NPZ outputs are written to ./outputs_study/

 UK English throughout. No commercial FEA. No external optimisation packages.
================================================================================
"""

from __future__ import annotations
import os
import csv
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ----------------------------------------------------------------------------
# GLOBAL CONFIGURATION  (edit here)
# ----------------------------------------------------------------------------
CONFIG = {
    "benchmark": "mbb",       # "mbb" or "cantilever"
    "nelx": 100,
    "nely": 40,
    "volfrac": 0.4,
    "penal": 3.0,
    "rmin": 2.0,
    "maxiter": 160,
    "seeds": [0, 1, 2, 3],    # raise to 20+ for the final paper
    "optimisers": {
        "OC":  {},
        "GD":  {"lr": 0.5},
        "MMA": {},
    },
    "outdir": "outputs_study",
}

PALETTE = {"OC": "#1f77b4", "GD": "#d62728", "MMA": "#2ca02c", "Adam": "#9467bd"}


# ============================================================================
# 1. SIMP PHYSICS
# ============================================================================
def element_stiffness(nu=0.3):
    k = np.array([0.5 - nu/6, 0.125 + nu/8, -0.25 - nu/12, -0.125 + 3*nu/8,
                  -0.25 + nu/12, -0.125 - nu/8, nu/6, 0.125 - 3*nu/8])
    KE = 1/(1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


class DensityFilter:
    def __init__(self, nelx, nely, rmin):
        nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
        iH = np.zeros(nfilter, dtype=int)
        jH = np.zeros(nfilter, dtype=int)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i*nely + j
                kk1 = int(max(i-(np.ceil(rmin)-1), 0))
                kk2 = int(min(i+np.ceil(rmin), nelx))
                ll1 = int(max(j-(np.ceil(rmin)-1), 0))
                ll2 = int(min(j+np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for ll in range(ll1, ll2):
                        col = k*nely + ll
                        fac = rmin - np.sqrt((i-k)**2 + (j-ll)**2)
                        iH[cc] = row; jH[cc] = col; sH[cc] = max(0.0, fac)
                        cc += 1
        self.H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])),
                            shape=(nelx*nely, nelx*nely)).tocsr()
        self.Hs = np.asarray(self.H.sum(axis=1)).flatten()

    def filter_density(self, x):
        return np.asarray(self.H @ x) / self.Hs

    def filter_sensitivity(self, dc):
        return np.asarray(self.H @ (dc / self.Hs))


class TopOptProblem:
    def __init__(self, nelx, nely, volfrac, penal=3.0, rmin=1.5,
                 nu=0.3, emin=1e-9, emax=1.0, bc="mbb"):
        self.nelx, self.nely = nelx, nely
        self.volfrac, self.penal, self.rmin = volfrac, penal, rmin
        self.nu, self.emin, self.emax, self.bc = nu, emin, emax, bc
        self.ndof = 2*(nelx+1)*(nely+1)
        self.KE = element_stiffness(nu)
        self.filt = DensityFilter(nelx, nely, rmin)
        self._build_edof()
        self._build_bc()

    def _build_edof(self):
        nelx, nely = self.nelx, self.nely
        self.edofMat = np.zeros((nelx*nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = elx*nely + ely
                n1 = (nely+1)*elx + ely
                n2 = (nely+1)*(elx+1) + ely
                self.edofMat[el, :] = [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                                       2*n2, 2*n2+1, 2*n1, 2*n1+1]
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten().astype(int)
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten().astype(int)

    def _build_bc(self):
        nelx, nely, ndof = self.nelx, self.nely, self.ndof
        self.f = np.zeros((ndof, 1))
        dofs = np.arange(ndof)
        if self.bc == "mbb":
            self.f[1, 0] = -1.0
            fixed = np.union1d(np.arange(0, 2*(nely+1), 2), [ndof-1])
        elif self.bc == "cantilever":
            self.f[2*(nelx+1)*(nely+1) - nely - 1, 0] = -1.0
            fixed = np.arange(0, 2*(nely+1))
        else:
            raise ValueError(self.bc)
        self.free = np.setdiff1d(dofs, fixed)

    def analyse(self, xPhys):
        E = self.emin + xPhys**self.penal*(self.emax - self.emin)
        sK = ((self.KE.flatten()[np.newaxis]).T*E).flatten(order="F")
        K = coo_matrix((sK, (self.iK, self.jK)),
                       shape=(self.ndof, self.ndof)).tocsc()
        K = (K + K.T)/2
        u = np.zeros((self.ndof, 1))
        u[self.free, 0] = spsolve(K[self.free, :][:, self.free],
                                  self.f[self.free, 0])
        ce = (np.dot(u[self.edofMat].reshape(self.nelx*self.nely, 8), self.KE)
              * u[self.edofMat].reshape(self.nelx*self.nely, 8)).sum(1)
        dE = self.penal*xPhys**(self.penal-1)*(self.emax - self.emin)
        c = (E*ce).sum()
        dc = -dE*ce
        return c, dc, xPhys.mean()


# ============================================================================
# 2. MMA  (Svanberg, single volume constraint)
# ============================================================================
class MMAState:
    def __init__(self, n):
        self.xold1 = self.xold2 = self.low = self.upp = None
        self.n = n


def mma_sub(x, dfdx, gx, dgdx, xmin, xmax, state,
            move=0.5, asyinit=0.5, asyincr=1.2, asydecr=0.7):
    n = len(x); xval = x.copy(); eeen = np.ones(n)
    if state.low is None or state.xold2 is None:
        state.low = xval - asyinit*(xmax - xmin)
        state.upp = xval + asyinit*(xmax - xmin)
    else:
        zzz = (xval - state.xold1)*(state.xold1 - state.xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr; factor[zzz < 0] = asydecr
        state.low = xval - factor*(state.xold1 - state.low)
        state.upp = xval + factor*(state.upp - state.xold1)
        state.low = np.clip(state.low, xval - 10*(xmax-xmin), xval - 0.01*(xmax-xmin))
        state.upp = np.clip(state.upp, xval + 0.01*(xmax-xmin), xval + 10*(xmax-xmin))
    alfa = np.maximum.reduce([xmin, state.low + 0.1*(xval-state.low),
                              xval - move*(xmax-xmin)])
    beta = np.minimum.reduce([xmax, state.upp - 0.1*(state.upp-xval),
                              xval + move*(xmax-xmin)])
    ux1 = state.upp - xval; xl1 = xval - state.low
    p0 = np.maximum(dfdx, 0)*ux1**2; q0 = np.maximum(-dfdx, 0)*xl1**2
    p1 = np.maximum(dgdx, 0)*ux1**2; q1 = np.maximum(-dgdx, 0)*xl1**2
    b1 = (p1/ux1 + q1/xl1).sum() - gx

    def primal(lam):
        plam = np.maximum(p0 + lam*p1, 1e-30)
        qlam = np.maximum(q0 + lam*q1, 1e-30)
        xn = (np.sqrt(plam)*state.low + np.sqrt(qlam)*state.upp) / \
             (np.sqrt(plam) + np.sqrt(qlam))
        return np.clip(xn, alfa, beta)

    def dual_grad(lam):
        xn = primal(lam)
        return (p1/(state.upp - xn) + q1/(xn - state.low)).sum() - b1

    if dual_grad(0.0) < 0:
        xnew = primal(0.0)
    else:
        lo, hi = 0.0, 1e9
        for _ in range(80):
            mid = 0.5*(lo+hi)
            if dual_grad(mid) > 0: lo = mid
            else: hi = mid
        xnew = primal(0.5*(lo+hi))
    state.xold2 = state.xold1
    state.xold1 = xval.copy()
    return xnew


# ============================================================================
# 3. OPTIMISERS  (one interface: step(x, dc, dv) -> x_new)
# ============================================================================
def _sigmoid(z):
    return 1.0/(1.0 + np.exp(-np.clip(z, -50, 50)))

Z_CLAMP = 25.0


def _volume_map(z, target_vol):
    l1, l2 = -50.0, 50.0
    for _ in range(80):
        b = 0.5*(l1+l2)
        if _sigmoid(z+b).mean() > target_vol: l2 = b
        else: l1 = b
    return _sigmoid(z + 0.5*(l1+l2))


class OCOptimiser:
    name = "OC"
    def __init__(self, problem, move=0.2, eta=0.5):
        self.p, self.move, self.eta = problem, move, eta
    def step(self, x, dc, dv):
        l1, l2 = 1e-9, 1e9; vol = self.p.volfrac; xnew = x.copy()
        while (l2-l1)/(0.5*(l1+l2)) > 1e-3:
            lmid = 0.5*(l1+l2)
            be = np.maximum(0.0, -dc/(dv*lmid))
            xnew = np.clip(np.clip(x*np.power(be, self.eta),
                                   np.maximum(0, x-self.move),
                                   np.minimum(1, x+self.move)), 0, 1)
            if xnew.mean() > vol: l1 = lmid
            else: l2 = lmid
        return xnew


class _LatentBase:
    def __init__(self, problem):
        self.p = problem; self.z = None; self._rng = None
    def _init_latent(self, x):
        xc = np.clip(x, 1e-4, 1-1e-4)
        self.z = np.log(xc/(1-xc))
        if self._rng is not None:
            self.z += 1e-2*(self._rng.random(self.z.shape)-0.5)
    def _finish(self):
        self.z = np.clip(self.z, -Z_CLAMP, Z_CLAMP)
        return _volume_map(self.z, self.p.volfrac)


class GDOptimiser(_LatentBase):
    name = "GD"
    def __init__(self, problem, lr=0.5):
        super().__init__(problem); self.lr = lr
    def step(self, x, dc, dv):
        if self.z is None: self._init_latent(x)
        g = dc*(x*(1-x))
        g = g/(np.sqrt(np.mean(g**2))+1e-12)
        self.z = self.z - self.lr*g
        return self._finish()


class MMAOptimiser:
    name = "MMA"
    def __init__(self, problem, move=0.2):
        self.p, self.move, self.state = problem, move, None
    def step(self, x, dc, dv):
        if self.state is None: self.state = MMAState(len(x))
        gx = x.mean() - self.p.volfrac
        xnew = mma_sub(x, dc, gx, dv, np.zeros(len(x)), np.ones(len(x)),
                       self.state, move=self.move)
        return np.clip(xnew, 0, 1)


OPTIMISERS = {"OC": OCOptimiser, "GD": GDOptimiser, "MMA": MMAOptimiser}


# ============================================================================
# 4. DRIVER  (fixed horizon, post-hoc convergence)
# ============================================================================
def detect_convergence(compliance, window=10, tol=1e-3):
    c = np.asarray(compliance)
    for i in range(window, len(c)):
        if abs(c[i-window]-c[i])/(abs(c[i-window])+1e-12) < tol:
            return i
    return len(c)


def run_optimisation(problem, name, maxiter=160, seed=0, x0=None, opt_kwargs=None):
    rng = np.random.default_rng(seed)
    n = problem.nelx*problem.nely
    if x0 == "random":
        x = np.clip(problem.volfrac + 0.1*(rng.random(n)-0.5), 0.01, 0.99)
        x *= problem.volfrac/x.mean(); x = np.clip(x, 0.01, 0.99)
    elif x0 is None:
        x = np.full(n, problem.volfrac)
    else:
        x = x0.copy()
    opt = OPTIMISERS[name](problem, **(opt_kwargs or {}))
    if hasattr(opt, "_rng"): opt._rng = rng
    xPhys = problem.filt.filter_density(x)
    hist = {"compliance": [], "change": [], "vol": [], "grayness": []}
    for it in range(maxiter):
        c, dc, vol = problem.analyse(xPhys)
        dv = np.ones(n)/n
        dc = problem.filt.filter_sensitivity(dc)
        dv = problem.filt.filter_sensitivity(dv)
        xold = x.copy()
        x = opt.step(x, dc, dv)
        xPhys = problem.filt.filter_density(x)
        hist["compliance"].append(float(c))
        hist["change"].append(float(np.abs(x-xold).max()))
        hist["vol"].append(float(vol))
        hist["grayness"].append(float(4*np.mean(xPhys*(1-xPhys))))
    return {"optimiser": name, "seed": seed, "xPhys": xPhys,
            "compliance": hist["compliance"][-1],
            "converged_at": detect_convergence(hist["compliance"]),
            "grayness": hist["grayness"][-1],
            "final_volume": float(xPhys.mean()), "history": hist}


# ============================================================================
# 5. METRICS
# ============================================================================
def binary_iou(xa, xb, t=0.5):
    a, b = xa >= t, xb >= t
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/u) if u else 1.0

def density_rmse(xa, xb):
    return float(np.sqrt(np.mean((xa-xb)**2)))

def hole_count(xPhys, nelx, nely, t=0.5):
    void = (xPhys.reshape(nelx, nely) < t)
    seen = np.zeros_like(void, dtype=bool); count = 0
    for i in range(nelx):
        for j in range(nely):
            if void[i, j] and not seen[i, j]:
                count += 1; stack = [(i, j)]; seen[i, j] = True
                while stack:
                    ci, cj = stack.pop()
                    for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                        ni, nj = ci+di, cj+dj
                        if 0<=ni<nelx and 0<=nj<nely and void[ni,nj] and not seen[ni,nj]:
                            seen[ni,nj]=True; stack.append((ni,nj))
    return count


# ============================================================================
# 6. EXPERIMENT
# ============================================================================
def run_experiment(cfg):
    os.makedirs(cfg["outdir"], exist_ok=True)
    prob = TopOptProblem(cfg["nelx"], cfg["nely"], cfg["volfrac"],
                         penal=cfg["penal"], rmin=cfg["rmin"], bc=cfg["benchmark"])
    results = []
    for name, kw in cfg["optimisers"].items():
        for seed in cfg["seeds"]:
            r = run_optimisation(prob, name, maxiter=cfg["maxiter"],
                                 seed=seed, x0="random", opt_kwargs=kw)
            results.append(r)
            print(f"{name:4s} seed {seed}: c={r['compliance']:8.3f} "
                  f"conv@{r['converged_at']:3d} gray={r['grayness']:.3f} "
                  f"vol={r['final_volume']:.3f}")
    return prob, results


# ============================================================================
# 7. VISUALISATION SUITE
# ============================================================================
def fig_convergence(results, cfg):
    """Per-iteration compliance for every run, grouped by optimiser, with legend."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    seen = set()
    for r in results:
        name = r["optimiser"]
        lbl = name if name not in seen else None
        seen.add(name)
        ax1.plot(r["history"]["compliance"], color=PALETTE[name], alpha=0.55,
                 lw=1.3, label=lbl)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Compliance (J)")
    ax1.set_title("Compliance convergence (all seeds)")
    ax1.set_yscale("log"); ax1.legend(title="Optimiser"); ax1.grid(alpha=0.3)
    seen = set()
    for r in results:
        name = r["optimiser"]
        lbl = name if name not in seen else None
        seen.add(name)
        ax2.plot(r["history"]["compliance"], color=PALETTE[name], alpha=0.55,
                 lw=1.3, label=lbl)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Compliance (J)")
    ax2.set_title("Compliance convergence (linear, zoomed)")
    finals = [r["compliance"] for r in results]
    ax2.set_ylim(min(finals)*0.9, np.percentile(finals, 75)*1.6)
    ax2.legend(title="Optimiser"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(cfg["outdir"], "fig1_convergence.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_change_grayness(results, cfg):
    """Design change and grayness vs iteration, mean +/- band per optimiser."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for metric, ax, ttl, yl in [
            ("change", ax1, "Max design change per iteration", "max |Δx|"),
            ("grayness", ax2, "Grayness (discreteness) vs iteration",
             "grayness  4·mean[x(1−x)]")]:
        for name in cfg["optimisers"]:
            runs = [r["history"][metric] for r in results
                    if r["optimiser"] == name]
            L = min(len(s) for s in runs)
            arr = np.array([s[:L] for s in runs])
            mean = arr.mean(0); std = arr.std(0)
            it = np.arange(L)
            ax.plot(it, mean, color=PALETTE[name], lw=2, label=name)
            ax.fill_between(it, mean-std, mean+std, color=PALETTE[name], alpha=0.18)
        ax.set_xlabel("Iteration"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(title="Optimiser"); ax.grid(alpha=0.3)
    ax1.set_yscale("log")
    fig.tight_layout()
    p = os.path.join(cfg["outdir"], "fig2_change_grayness.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_distributions(results, cfg):
    """Box + strip of final compliance, and bar of convergence iteration."""
    names = list(cfg["optimisers"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    data = [[r["compliance"] for r in results if r["optimiser"] == n] for n in names]
    bp = ax1.boxplot(data, tick_labels=names, patch_artist=True, widths=0.5)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(PALETTE[n]); patch.set_alpha(0.4)
    for i, n in enumerate(names):
        ys = data[i]; xs = np.random.normal(i+1, 0.04, len(ys))
        ax1.scatter(xs, ys, color=PALETTE[n], s=28, zorder=3, edgecolor="k", lw=0.4)
    ax1.set_ylabel("Final compliance (J)")
    ax1.set_title("Final-compliance distribution across seeds")
    ax1.grid(alpha=0.3, axis="y")
    conv_means = [np.mean([r["converged_at"] for r in results if r["optimiser"]==n])
                  for n in names]
    conv_std = [np.std([r["converged_at"] for r in results if r["optimiser"]==n])
                for n in names]
    ax2.bar(names, conv_means, yerr=conv_std, capsize=6,
            color=[PALETTE[n] for n in names], alpha=0.7, edgecolor="k")
    ax2.set_ylabel("Convergence iteration")
    ax2.set_title("Iterations to convergence (mean ± s.d.)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(cfg["outdir"], "fig3_distributions.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_agreement(prob, results, cfg):
    """Heatmaps: cross-optimiser IoU (seed 0) and within-optimiser reproducibility."""
    names = list(cfg["optimisers"])
    seed0 = {n: next(r["xPhys"] for r in results
                     if r["optimiser"]==n and r["seed"]==cfg["seeds"][0])
             for n in names}
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = binary_iou(seed0[a], seed0[b])
    repro = []
    for n in names:
        ds = [r["xPhys"] for r in results if r["optimiser"] == n]
        ious = [binary_iou(ds[i], ds[j]) for i in range(len(ds))
                for j in range(i+1, len(ds))]
        repro.append(np.mean(ious))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    im = ax1.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names)
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax1.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                     color="white" if M[i,j] < 0.6 else "black", fontsize=11)
    ax1.set_title("Cross-optimiser design agreement (IoU, seed 0)")
    fig.colorbar(im, ax=ax1, fraction=0.046, label="IoU")
    bars = ax2.bar(names, repro, color=[PALETTE[n] for n in names],
                   alpha=0.75, edgecolor="k")
    ax2.set_ylim(0, 1); ax2.set_ylabel("Mean pairwise IoU across seeds")
    ax2.set_title("Within-optimiser reproducibility\n(higher = more repeatable)")
    ax2.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, repro):
        ax2.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    p = os.path.join(cfg["outdir"], "fig4_agreement.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_design_grid(prob, results, cfg):
    """Final design of every run, rows = optimiser, cols = seed."""
    names = list(cfg["optimisers"]); seeds = cfg["seeds"]
    fig, axes = plt.subplots(len(names), len(seeds),
                             figsize=(3.2*len(seeds), 1.5*len(names)))
    axes = np.atleast_2d(axes)
    for i, n in enumerate(names):
        for j, s in enumerate(seeds):
            r = next(rr for rr in results if rr["optimiser"]==n and rr["seed"]==s)
            img = r["xPhys"].reshape(prob.nelx, prob.nely).T
            ax = axes[i, j]
            ax.imshow(-img, cmap="gray", vmin=-1, vmax=0)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0: ax.set_ylabel(n, fontsize=12, color=PALETTE[n],
                                     fontweight="bold")
            if i == 0: ax.set_title(f"seed {s}", fontsize=10)
            ax.text(0.97, 0.06, f"c={r['compliance']:.0f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
    fig.suptitle(f"Final designs  ({cfg['benchmark']} benchmark, "
                 f"vol={cfg['volfrac']})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(cfg["outdir"], "fig5_design_grid.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_compliance_vs_grayness(results, cfg):
    """Scatter: performance (compliance) vs discreteness (grayness)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in cfg["optimisers"]:
        cs = [r["compliance"] for r in results if r["optimiser"] == n]
        gs = [r["grayness"] for r in results if r["optimiser"] == n]
        ax.scatter(cs, gs, color=PALETTE[n], s=90, alpha=0.8,
                   edgecolor="k", lw=0.5, label=n)
    ax.set_xlabel("Final compliance (J)  — lower is better")
    ax.set_ylabel("Grayness — lower is a cleaner 0/1 design")
    ax.set_title("Performance vs manufacturability trade-off")
    ax.legend(title="Optimiser"); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(cfg["outdir"], "fig6_compliance_vs_grayness.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def write_csv(prob, results, cfg):
    rows = []
    for r in results:
        rows.append({
            "optimiser": r["optimiser"], "seed": r["seed"],
            "final_compliance": round(r["compliance"], 4),
            "converged_at": r["converged_at"],
            "grayness": round(r["grayness"], 4),
            "final_volume": round(r["final_volume"], 4),
            "holes": hole_count(r["xPhys"], prob.nelx, prob.nely),
        })
    p = os.path.join(cfg["outdir"], "results.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return p


# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    cfg = CONFIG
    print("="*70)
    print(" Topology-optimisation optimiser comparison study")
    print(f" benchmark={cfg['benchmark']}  mesh={cfg['nelx']}x{cfg['nely']}  "
          f"seeds={cfg['seeds']}  maxiter={cfg['maxiter']}")
    print("="*70)
    prob, results = run_experiment(cfg)

    print("\nGenerating figures...")
    outs = [
        fig_convergence(results, cfg),
        fig_change_grayness(results, cfg),
        fig_distributions(results, cfg),
        fig_agreement(prob, results, cfg),
        fig_design_grid(prob, results, cfg),
        fig_compliance_vs_grayness(results, cfg),
        write_csv(prob, results, cfg),
    ]
    np.savez_compressed(os.path.join(cfg["outdir"], "designs.npz"),
                        **{f"{r['optimiser']}_seed{r['seed']}": r["xPhys"]
                           for r in results})

    print("\n--- summary ---")
    for n in cfg["optimisers"]:
        cs = [r["compliance"] for r in results if r["optimiser"] == n]
        print(f"{n:4s}: compliance mean={np.mean(cs):.2f} sd={np.std(cs):.2f}")
    print("\nOutputs written to ./" + cfg["outdir"] + "/")
    for o in outs:
        print("  " + o)


if __name__ == "__main__":
    main()
