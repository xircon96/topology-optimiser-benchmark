"""
topopt_core.py
--------------
Density-based (SIMP) topology optimisation for 2D compliance minimisation,
written so that the optimiser update rule is fully decoupled from the physics.

The physics (finite element analysis + sensitivity analysis) is identical for
every optimiser. Only the update step differs. This is the design that lets us
study optimiser-induced non-uniqueness fairly: same gradient, same filter, same
penalty, only the update rule changes.

Based on the classic 88-line / 99-line educational SIMP formulation
(Sigmund 2001; Andreassen et al. 2011), re-implemented in NumPy/SciPy.

UK English used throughout. No commercial FEA required.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Element stiffness matrix for a bilinear quadrilateral (unit square element)
# ---------------------------------------------------------------------------
def element_stiffness(nu: float = 0.3) -> np.ndarray:
    """Return the 8x8 stiffness matrix for a unit-square Q4 element, E = 1."""
    k = np.array([
        0.5 - nu / 6.0,
        0.125 + nu / 8.0,
        -0.25 - nu / 12.0,
        -0.125 + 3.0 * nu / 8.0,
        -0.25 + nu / 12.0,
        -0.125 - nu / 8.0,
        nu / 6.0,
        0.125 - 3.0 * nu / 8.0,
    ])
    KE = 1.0 / (1.0 - nu ** 2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE


# ---------------------------------------------------------------------------
# Density filter (Bruns & Tortorelli style, linear hat weights)
# ---------------------------------------------------------------------------
class DensityFilter:
    """Precomputes the sparse filter matrix H and its row sums Hs."""

    def __init__(self, nelx: int, nely: int, rmin: float):
        self.nelx = nelx
        self.nely = nely
        self.rmin = rmin
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter, dtype=int)
        jH = np.zeros(nfilter, dtype=int)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for ll in range(ll1, ll2):
                        col = k * nely + ll
                        fac = rmin - np.sqrt((i - k) ** 2 + (j - ll) ** 2)
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc += 1
        self.H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])),
                            shape=(nelx * nely, nelx * nely)).tocsr()
        self.Hs = np.asarray(self.H.sum(axis=1)).flatten()

    def filter_density(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.H @ x) / self.Hs

    def filter_sensitivity(self, dc: np.ndarray) -> np.ndarray:
        return np.asarray(self.H @ (dc / self.Hs))


# ---------------------------------------------------------------------------
# The problem definition: mesh, loads, supports, SIMP parameters
# ---------------------------------------------------------------------------
class TopOptProblem:
    """Holds everything that is identical across optimisers."""

    def __init__(self, nelx, nely, volfrac, penal=3.0, rmin=1.5,
                 nu=0.3, emin=1e-9, emax=1.0, bc="mbb"):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.nu = nu
        self.emin = emin
        self.emax = emax
        self.bc = bc

        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.KE = element_stiffness(nu)
        self.filt = DensityFilter(nelx, nely, rmin)

        self._build_edof()
        self._build_bc()

    def _build_edof(self):
        nelx, nely = self.nelx, self.nely
        self.edofMat = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = elx * nely + ely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = np.array([
                    2 * n1 + 2, 2 * n1 + 3,
                    2 * n2 + 2, 2 * n2 + 3,
                    2 * n2, 2 * n2 + 1,
                    2 * n1, 2 * n1 + 1,
                ])
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten().astype(int)
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten().astype(int)

    def _build_bc(self):
        """Define load vector f and fixed dofs for the chosen benchmark."""
        nelx, nely, ndof = self.nelx, self.nely, self.ndof
        self.f = np.zeros((ndof, 1))
        if self.bc == "mbb":
            # MBB beam: downward load at top-left, symmetry on left edge,
            # roller at bottom-right corner.
            self.f[1, 0] = -1.0
            dofs = np.arange(ndof)
            fixed = np.union1d(
                np.arange(0, 2 * (nely + 1), 2),          # left edge x-fixed
                np.array([ndof - 1]),                     # bottom-right y-fixed
            )
            self.free = np.setdiff1d(dofs, fixed)
        elif self.bc == "cantilever":
            # Cantilever: left edge fully clamped, point load at mid-right.
            self.f[2 * (nelx + 1) * (nely + 1) - nely - 1, 0] = -1.0
            dofs = np.arange(ndof)
            fixed = np.arange(0, 2 * (nely + 1))
            self.free = np.setdiff1d(dofs, fixed)
        else:
            raise ValueError(f"Unknown bc '{self.bc}'")

    # -- physics: solve FE, return compliance and its sensitivity -----------
    def analyse(self, xPhys: np.ndarray):
        """Given physical densities, return (compliance, dc, volume)."""
        E = self.emin + xPhys ** self.penal * (self.emax - self.emin)
        sK = ((self.KE.flatten()[np.newaxis]).T * E).flatten(order="F")
        K = coo_matrix((sK, (self.iK, self.jK)),
                       shape=(self.ndof, self.ndof)).tocsc()
        K = (K + K.T) / 2.0
        u = np.zeros((self.ndof, 1))
        Kff = K[self.free, :][:, self.free]
        u[self.free, 0] = spsolve(Kff, self.f[self.free, 0])

        ce = (np.dot(u[self.edofMat].reshape(self.nelx * self.nely, 8),
                     self.KE) *
              u[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)
        dE = self.penal * xPhys ** (self.penal - 1) * (self.emax - self.emin)
        c = (E * ce).sum()
        dc = -dE * ce
        vol = xPhys.mean()
        return c, dc, vol
