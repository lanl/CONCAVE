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
    M[np.tril_indices(N,-1)] = M[np.triu_indices(N,1)].conj()
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
        bv = np.zeros(len(A))
        for (k,op) in enumerate(A.keys()):
            if np.sum(np.abs(A[op] - A[op].conj().T)) > 1e-8:
                raise Exception("A appears not to be Hermitian")
            Av[:,k] = _hpack(A[op])
            bv[k] = b[op]

        svdU, svdS, svdVh = np.linalg.svd(Av.T)
        rank = np.sum(svdS > 1e-8)
        Mv = svdVh[rank:].T

        for n in range(N):
            m.append(_hunpack(Mv[:,n]))

        for (n,M) in enumerate(m):
            c.append(np.trace(C @ M))

        M0v = svdVh[:rank].conj().T @ np.diag(1/svdS[:rank]) @ svdU[:,:rank].conj().T @ bv
        M0 = _hunpack(M0v)
        if True:
            for op in A.keys():
                # TODO these should all match...
                print(np.trace(A[op] @ M0), "  ", b[op])
            exit(0)
        return SemidefiniteProgram(M0, m, c)

class InteriorPointSolver:
    def __init__(self, sdp):
        self.M = sdp.M0

    def phase1(self):
        pass

    def solve(self):
        self.phase1()
        pass

