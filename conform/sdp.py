#!/usr/bin/env python

from functools import partial
import itertools
import sys

import numpy as np
import numpy.random as nr

SLACK = 0

if SLACK != 0:
    print("Warning: non-zero slack in use!", file=sys.stderr)

# A packing/unpacking of Hermitian matrices, which preserves the inner product.
def _hpack(M):
    N = M.shape[0]
    v = np.zeros(N*N)
    v[:N] = M[np.diag_indices(N)].real
    v[N:(N*(N+1))//2] = M[np.triu_indices(N,1)].real*np.sqrt(2)
    v[(N*(N+1))//2:] = M[np.triu_indices(N,1)].imag*np.sqrt(2)
    return v
   
def _hunpack(v):
    N = round(np.sqrt(len(v)))
    assert len(v) == N*N
    M = np.zeros((N,N), dtype=np.complex128)
    M[np.diag_indices(N)] = v[:N]
    M[np.triu_indices(N,1)] += v[N:(N*(N+1))//2]/np.sqrt(2)
    M[np.triu_indices(N,1)] += 1j*v[(N*(N+1))//2:]/np.sqrt(2)
    M += M.conj().T
    M[np.diag_indices(N)] /= 2
    return M

def _mpack(M):
    N = M.shape[0]
    Mf = M.flatten()
    v = np.zeros(2*N*N)
    v[:N] = Mf.real
    v[N:] = Mf.imag
    return v

def _munpack(v):
    N = round(np.sqrt(len(v)/2))
    assert len(v) == 2*N*N
    M = np.zeros((N,N), dtype=np.complex128)
    M += v[:N].reshape((N,N))
    M += 1j*v[N:].reshape((N,N))
    return M

class SemidefiniteProgram:
    def __init__(self, M0, m, c, const=0):
        #M0, m, c = self._reduce(M0, m, c)
        self.K = len(m)
        assert len(c) == self.K
        self.M0 = np.array(M0)
        self.m = np.array(m)
        self.c = np.array(c)
        self.N = M0.shape[0]
        self.const = const

    @staticmethod
    def _reduce(M0, m, c):
        M0 = np.array(M0)
        m = np.array(m)
        c = np.array(c)

        K = m.shape[0]
        N = M0.shape[0]

        # M0 is (N,N); m is (K,N,N). Both are Hermitian.

        raise Exception("not yet implemented")
        return M0, m, c

    def initial(self):
        return np.zeros((self.K,))

    @staticmethod
    def by_constraints(C, A, b, const=0):
        N = list(A.values())[0].shape[0]

        m = []
        c = []

        # Hermitize A
        for (k,op) in enumerate(A.keys()):
            A[op] = (A[op] + A[op].conj().T)/2

        Av = np.zeros((N*N, len(A)))
        bv = np.zeros(len(A))
        for (k,op) in enumerate(A.keys()):
            assert np.isreal(b[op])
            Av[:,k] = _hpack(A[op])
            bv[k] = b[op]

        Av = Av.T

        svdU, svdS, svdVh = np.linalg.svd(Av)
        rank = np.sum(svdS > 1e-8)
        Mv = svdVh[rank:].T

        K = Mv.shape[1]
        for k in range(K):
            m.append(_hunpack(Mv[:,k]))

        for (k,M) in enumerate(m):
            c.append(np.trace(C @ M).real)

        M0v = svdVh[:rank].conj().T @ np.diag(1/svdS[:rank]) @ svdU[:,:rank].conj().T @ bv
        M0 = _hunpack(M0v)
        return SemidefiniteProgram(M0, m, c, const)

    def _matrix(self, y):
        return self.M0 + np.einsum("iab,i->ab", self.m, y) + SLACK*np.identity(self.N)

    def feasible(self, y):
        M = self._matrix(y)
        mv = np.min(np.linalg.eigvalsh(M))
        return mv > 0

    def objective(self, y, *, differentiate=False):
        r = np.dot(self.c, y)
        if differentiate:
            return r + self.const, self.c
        return r + self.const

    def barrier(self, y, *, differentiate=False):
        M = self._matrix(y)
        vals = np.linalg.eigvalsh(M)
        if np.any(vals <= 0):
            return np.inf
        ld = np.sum(np.log(vals))
        if differentiate:
            Minv = np.linalg.inv(M)
            g = np.einsum("ij,aji->a", Minv, self.m).real
            h = -np.einsum("ij,ajk,kl,bli->ab", Minv, self.m, Minv, self.m, optimize=True).real
            return -ld, -g, -h
        return -ld

class _Phase1Program:
    def __init__(self, sdp):
        self.K = sdp.K+1
        self.sdp = sdp

    def initial(self):
        y = np.zeros((self.K,))
        M = self.sdp._matrix(y[1:])
        vs = np.linalg.eigvalsh(M)
        y[0] = 1-np.min(vs)
        return y

    def feasible(self, y):
        s, y = y[0], y[1:]
        M = self.sdp._matrix(y)
        vs = np.linalg.eigvalsh(M)
        return np.all(vs > -s)

    def objective(self, y, *, differentiate=False):
        if differentiate:
            g = np.zeros_like(y)
            g[0] = 1
            return y[0], g
        return y[0]

    def barrier(self, y, *, differentiate=False):
        s, y = y[0], y[1:]
        R = 1e-2
        reg = R * np.sum(y*y)/2
        M = self.sdp._matrix(y)
        M += s*np.identity(self.sdp.N)
        vs = np.linalg.eigvalsh(M)
        neg = np.min(vs) <= 0
        if differentiate:
            if neg:
                raise Exception("Requested to differentiate infinity")
            ld = np.sum(np.log(vs).real)
            Minv = np.linalg.inv(M)
            g_ = np.einsum("ij,aji->a", Minv, self.sdp.m)
            h_ = -np.einsum("ij,ajk,kl,bli->ab", Minv, self.sdp.m, Minv, self.sdp.m, optimize=True)
            g = np.zeros((self.K,))
            g[1:] = g_.real
            g[0] = np.trace(Minv).real
            h = np.zeros((self.K,self.K))
            h[1:,1:] = h_.real
            h[0,0] = -np.einsum("ij,ji", Minv, Minv).real
            h[0,1:] = -np.einsum("ij,ajk,ki->a", Minv, self.sdp.m, Minv, optimize=True).real
            h[1:,0] = h[0,1:]

            r = -ld+reg
            g = -g
            g[1:] += R*y
            h = -h
            h[1:,1:] += R*np.identity(self.sdp.K)
            return r, g, h
        if neg:
            return np.inf
        ld = np.sum(np.log(vs).real)
        # TODO logarithm unbounded above... why is the regulator necessary?
        return -ld + reg

def newton(loss, y, t, *, maxiter=1000):
    K = len(y)
    niter = 0
    while niter < maxiter:
        v, g, h = loss(y, t, differentiate=True)
        h += 1e-8 * np.identity(K)

        dy = -np.linalg.solve(h,g)

        # Check termination
        delta = np.dot(g, dy) / 4
        if np.linalg.norm(dy) < 1e-10 or np.abs(delta)/np.abs(v) < 1e-10:
            break

        # Backtracking line search
        alpha = 1.0
        m = np.dot(g, dy)
        yp = y + alpha * dy
        vp = loss(yp,t)
        while vp > v + 0.5 * alpha * m and alpha > 1e-30:
            alpha *= 0.5
            yp = y + alpha * dy
            vp = loss(yp,t)
        v = vp
        y = yp

        if alpha < 1e-30:
            break

        niter += 1

    return y, niter

class InteriorPointSolver:
    def __init__(self, sdp):
        K = sdp.K
        self.sdp = sdp
        self.y = sdp.initial()

    def solve(self, *, verbose=False):
        if not self.sdp.feasible(self.y):
            if verbose:
                print("Solving phase 1...")
            _phase1 = _Phase1Program(self.sdp)
            _solver = InteriorPointSolver(_phase1)
            _solver.solve(verbose=verbose)
            if verbose:
                print("  Phase 1 complete!")
            self.y = _solver.y[1:]
        else:
            if verbose:
                print("Initial point was feasible; solving...")
        if not self.sdp.feasible(self.y):
            raise Exception("No feasible initial point found")

        def loss(y, t, *, differentiate=False):
            if differentiate:
                obj, objg = self.sdp.objective(y, differentiate=True)
                bar, barg, h = self.sdp.barrier(y, differentiate=True)
                return obj + bar/t, objg+barg/t, h/t
            obj = self.sdp.objective(y)
            bar = self.sdp.barrier(y)
            return obj + bar/t

        t = 1e-3
        mu = 2.0
        eps = 1e-10

        while t < 1/eps:
            if verbose:
                print(f"{t} ", end='', flush=True)
            # Center
            t = mu*t
            self.y, niter = newton(loss, self.y, t, maxiter=300)
            if verbose:
                print(f"{self.sdp.objective(self.y)} {loss(self.y,t)} {niter}", flush=True)
        return self.sdp.objective(self.y)

