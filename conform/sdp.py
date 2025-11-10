#!/usr/bin/env python

import itertools

import jax
import jax.numpy as jnp

import numpy as np
import numpy.random as nr

def _hpack(M):
    N = M.shape[0]
    v = np.zeros(N*N)
    v[:(N*(N+1))//2] = M[np.triu_indices(N)].real
    v[(N*(N+1))//2:] = M[np.triu_indices(N,1)].imag
    return v
   
def _hunpack(v):
    N = round(np.sqrt(len(v)))
    assert len(v) == N*N
    M = np.zeros((N,N), dtype=np.complex128)
    M[np.triu_indices(N)] = v[:(N*(N+1))//2]
    M[np.triu_indices(N,1)] += 1j*v[(N*(N+1))//2:]
    return M

class SemidefiniteProgram:
    def __init__(self, M0, m, c):
        self.K = len(m)
        assert len(c) == self.K
        self.N = M0.shape[0]
        self.M0 = M0
        self.m = m
        self.c = c

    @staticmethod
    def by_constraints(C, A, b):
        N = list(A.values())[0].shape[0]

        m = []
        c = []

        Av = np.zeros((N*N, len(A)))
        for (k,M) in enumerate(A.values()):
            Av[:,k] = _hpack(M)

        svdU, svdS, svdVt = np.linalg.svd(Av.T)
        rank = np.sum(svdS > 1e-8)
        Mv = svdVt[rank:].T

        for n in range(N):
            m.append(_hunpack(Mv[:,n]))

        for (n,M) in enumerate(m):
            c.append(np.trace(C @ M))

        # TODO M0
        exit(0)
        return SemidefiniteProgram(M0, m, c)

class InteriorPointSolver:
    def __init__(self, sdp):
        pass

    def solve(self):
        pass

