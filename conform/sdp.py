#!/usr/bin/env python

from functools import partial
import itertools

import jax
import jax.debug as jdb
import jax.numpy as jnp
import jax.random as jr

import numpy as np
import numpy.random as nr

jax.config.update('jax_enable_x64', True)

# TODO
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_disable_jit", True)

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
    def __init__(self, M0, m, c):
        self.K = len(m)
        assert len(c) == self.K
        self.N = M0.shape[0]
        self.M0 = jnp.array(M0)
        self.m = jnp.array(m)
        self.c = jnp.array(c)

    def initial(self):
        return jnp.zeros((self.K,))

    @staticmethod
    def by_constraints(C, A, b):
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
            c.append(np.trace(C @ M))

        M0v = svdVh[:rank].conj().T @ np.diag(1/svdS[:rank]) @ svdU[:,:rank].conj().T @ bv
        M0 = _hunpack(M0v)
        return SemidefiniteProgram(M0, m, c)

    def _matrix(self):
        def f(y):
            return self.M0 + jnp.einsum("iab,i->ab", self.m, y)
        return f

    def feasible(self):
        @jax.jit
        def f(y):
            M = self._matrix()(y)
            mv = jnp.min(jnp.linalg.eigvalsh(M))
            return mv > 0
        return f

    def objective(self):
        def f(y, *, differentiate=False):
            #TODO differentiate
            return jnp.dot(self.c, y)
        return f

    def barrier(self):
        def f(y, *, differentiate=False):
            # TODO differentiate
            M = self._matrix()(y)
            _, ld = jnp.linalg.slogdet(M)
            return -ld
        return f

class _Phase1Program:
    def __init__(self, sdp):
        self.K = sdp.K+1
        self.sdp = sdp

    def initial(self):
        y = jnp.zeros((self.K,))
        matrix = self.sdp._matrix()
        M = matrix(y[1:])
        vs = jnp.linalg.eigvalsh(M)
        return y.at[0].set(1-jnp.min(vs))

    def feasible(self):
        feasible = self.sdp.feasible()
        matrix = self.sdp._matrix()
        @jax.jit
        def f(y):
            s, y = y[0], y[1:]
            M = matrix(y)
            vs = jnp.linalg.eigvalsh(M)
            return jnp.all(vs > -s)
        return f

    def objective(self, *, differentiate=False):
        #TODO differentiate
        def f(y):
            return y[0]
        return f

    def barrier(self):
        #TODO differentiate
        matrix = self.sdp._matrix()
        def f(y, *, differentiate=False):
            s, y = y[0], y[1:]
            M = matrix(y)
            M += s*jnp.identity(self.sdp.N)
            print(M)
            vs = jnp.linalg.eigvalsh(M)
            neg = jnp.min(vs) <= 0
            ld = jnp.sum(jnp.log(vs).real)
            return jax.lax.select(neg, jnp.inf, -ld)
        return f

def newton(loss, y, t):
    while True:
        v, g, h = loss(y, t, differentiate=True)
        # Compute gradient and hessian.
        #v, g = jax.value_and_grad(loss)(y,t)
        #h = jax.hessian(loss)(y,t)

        dy = -jnp.linalg.solve(h,g)

        # Check termination
        delta = jnp.dot(g, dy) / 4
        if jnp.linalg.norm(dy) < 1e-10 or jnp.abs(delta)/jnp.abs(v) < 1e-10:
            break

        # Backtracking line search
        alpha = 1.0
        m = jnp.dot(g, dy)
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

    return y

class InteriorPointSolver:
    def __init__(self, sdp):
        K = sdp.K
        self.sdp = sdp
        self.y = sdp.initial()

    def solve(self, *, verbose=True):
        feasible = self.sdp.feasible()
        if not feasible(self.y):
            _phase1 = _Phase1Program(self.sdp)
            _solver = InteriorPointSolver(_phase1)
            _solver.solve()
            self.y = _solver.y[1:]
        if not feasible(self.y):
            raise Exception("No feasible initial point found")

        objective = self.sdp.objective()
        barrier = self.sdp.barrier()

        def loss(y, t, *, differentiate=False):
            if differentiate:
                obj, objg = objective(y, differentiate=True)
                bar, barg, h = barrier(y, differentiate=True)
                return obj + bar/t, objg+barg/t, h/t
            obj = objective(y)
            bar = barrier(y)
            return obj + bar/t

        t = 1e-2
        mu = 1.5
        eps = 1e-10

        while t < 1/eps:
            print(t)
            # Center
            t = mu*t
            self.y = newton(loss, self.y, t)

