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
    pass

def test_phase1_objective_gradient():
    phase1 = _make_phase1(8,13)

def test_phase1_barrier_gradient():
    phase1 = _make_phase1(8,13)

def test_phase1_barrier_hessian():
    phase1 = _make_phase1(8,13)
