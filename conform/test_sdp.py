import numpy as np
import numpy.random as nr

import sdp

def test_hpack_cycle():
    for N in range(1,10):
        M = nr.normal(size=(N,N)) + 1j*nr.normal(size=(N,N))
        M = M.conj().T + M
        M2 = sdp._hunpack(sdp._hpack(M))
        assert np.sum(np.abs(M2-M)) < 1e-5

def test_hunpack_cycle():
    for N in range(1,10):
        v = nr.normal(size=N*N)
        v2 = sdp._hpack(sdp._hunpack(v))
        assert np.sum(np.abs(v2-v)) < 1e-5

def test_hpack_product():
    for N in range(1,11):
        u = nr.normal(size=N*N)
        v = nr.normal(size=N*N)
        ip1 = np.sum(u.conj()*v).real
        A = sdp._hunpack(u)
        B = sdp._hunpack(v)
        ip2 = np.trace(A.conj().T @ B)
        assert np.abs(ip1-ip2) < 1e-5

def test_sdp_objective_gradient():
    pass

def test_sdp_barrier_gradient():
    pass

def test_sdp_barrier_hessian():
    pass

def _make_phase1(N,K):
    M0 = nr.normal(size=(N,N)) + 1j*nr.normal(size=(N,N))
    M0 = M0 + M0.conj().T
    m = nr.normal(size=(K,N,N)) + 1j*nr.normal(size=(K,N,N))
    m += np.einsum("aij->aji", m.conj())
    c = nr.normal(size=(K,))
    prog = sdp.SemidefiniteProgram(M0, m, c)
    return sdp._Phase1Program(prog)

def test_phase1_objective_gradient():
    K = 13
    phase1 = _make_phase1(8,K)
    y = nr.normal(size=K)
    obj0, grad = phase1.objective(y, differentiate=True)
    eps = 1e-5
    for k in range(K):
        yk = y.copy()
        yk[k] += eps
        objk = phase1.objective(yk)
        d = (objk-obj0)/eps
        assert np.abs(d - grad[k]) < 1e-4

def test_phase1_barrier_gradient():
    K = 13
    phase1 = _make_phase1(8,K)
    y = nr.normal(size=K)
    bar0, grad = phase1.barrier(y, differentiate=True)
    eps = 1e-5
    for k in range(K):
        yk = y.copy()
        yk[k] += eps
        bark = phase1.barrier(yk)
        d = (bark-bar0)/eps
        assert np.abs(d - grad[k]) < 1e-4

def test_phase1_barrier_hessian():
    K = 13
    phase1 = _make_phase1(8,K)
