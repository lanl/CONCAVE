#!/usr/bin/env python

import itertools

import jax
import jax.numpy as jnp

import numpy as np
import numpy.random as nr

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
        self.M0 = M0
        self.m = m
        self.c = c

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
        # TODO I don't think this packing preserves the inner product... the
        # diagonal is weighted differently, right?

        Av = Av.T

        svdU, svdS, svdVh = np.linalg.svd(Av)
        rank = np.sum(svdS > 1e-8)
        Mv = svdVh[rank:].T

        for n in range(N):
            m.append(_hunpack(Mv[:,n]))

        for (n,M) in enumerate(m):
            c.append(np.trace(C @ M))

        M0v = svdVh[:rank].conj().T @ np.diag(1/svdS[:rank]) @ svdU[:,:rank].conj().T @ bv
        M0 = _hunpack(M0v)
        print("VIOLATION: ", np.sum(np.abs(Av @ M0v - bv)))
        print(Av @ M0v)
        if True:
            print("======")
            for op in A.keys():
                for M in m:
                    if np.abs(np.trace(A[op]@M)) > 1e-8:
                        print("OH NO!")
                        print(np.trace(A[op]@M))
                # TODO these should all match...
                if np.sum(np.abs(np.trace(A[op] @ M0) - b[op])) > 1e-5:
                    print(op)
                    print(M0)
                    print(A[op])
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

