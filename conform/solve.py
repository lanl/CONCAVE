#!/usr/bin/env python

import re
import subprocess
import sys

import numpy as np

from sdp import *

GLOBALS = {}

class Term:
    def __init__(self, idx, c=1, op=''):
        self.c = c
        self.idx = idx
        self.op = op

    def __repr__(self):
        if self.idx is None:
            return f"{self.c}*[{self.op}]"
        else:
            return f"{self.c}*M({self.idx})*[{self.op}]"

    @staticmethod
    def parse(s):
        ops = ''
        mcoef = None
        c = 1
        match = re.match(r'(?ms)([+-])\s+(.*)',s)
        if match.group(1) == '-':
            c *= -1
        for factor in match.group(2).split('*'):
            if m := re.match(r'oprod\(([^)]+)\)', factor):
                ops = m.group(1)
            elif m := re.match(r'M\(([0-9]+),([0-9]+)\)', factor):
                if mcoef is not None:
                    raise Exception("Nonlinear in M!")
                mcoef = (int(m.group(1)),int(m.group(2)))
            elif re.match(r'i_', factor):
                c *= 1j
            else:
                n = eval(factor.replace('^','**'), GLOBALS)
                c *= float(n)
        return Term(mcoef, c, ops)

def parse_expression(expr):
    # Split into terms. A term is a thing that begins ^\s+[+-].
    term_strings = re.findall(r'(?ms)\s+([+-].+?)(?=\s+(?:[+-]|\Z))' ,expr)
    terms = list(map(Term.parse, term_strings))
    return terms

def make_sdp(form, *, verbose=False):
    if type(form) == bytes:
        form = form.decode()
    # Extract the two expressions.
    ham_start = form.index('hamiltonian =') + len('hamiltonian =')
    ham_end = form.index(';', ham_start)
    ham = parse_expression(form[ham_start:ham_end])
    sos_start = form.index('sos =') + len('sos =')
    sos_end = form.index(';', sos_start)
    sos = parse_expression(form[sos_start:sos_end])

    #print(ham)
    #print(sos)

    N = 0
    ops = set()
    for term in sos:
        N = max(N, *term.idx)
        ops.add(term.op)
    N += 1

    # Check that there are no duplicates in ops. This is a symptom of a very
    # common sort of error in the underlying computer algebra.
    for op1 in ops:
        for op2 in ops:
            s1 = sorted(op1.split(','))
            s2 = sorted(op2.split(','))
            if op1 != op2 and s1 == s2:
                raise Exception(f"Duplicate found: {op1} and {op2}")

    c = 0.0
    C = np.zeros((N,N), dtype=np.complex128)
    A = {}
    b = {}
    for op in ops:
        A[op] = np.zeros((N,N), dtype=np.complex128)
        b[op] = 0.0

    for term in ham:
        if term.idx is not None:
            raise Exception('Coefficient appeared in Hamiltonian')
        if term.op == '':
            raise Exception('Constant term in Hamiltonian, not handled')
        b[term.op] += term.c
    for term in sos:
        if term.op == '':
            if term.idx is None:
                c += term.c
            else:
                C[term.idx] += term.c
        else:
            A[term.op][term.idx] += term.c

    if verbose:
        print("C:")
        print(C)
        print()
        for op in ops:
            print(f"b[{op}]: {b[op]}")
            print(f"A[{op}]:")
            print(A[op])
            print()

    return SemidefiniteProgram.by_constraints(C, A, b, c)

def solve(sdp, *, verbose=False):
    ipm = InteriorPointSolver(sdp)
    obj = ipm.solve(verbose=verbose)
    if verbose:
        print(f"y: {ipm.y}")
        M = sdp._matrix(ipm.y)
        print(f"M: {M}")
        print(f" eigenvalues: {np.linalg.eigvalsh(M)}")
    return -obj

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: ./solve.py (hydrogen|dihydrogen)')
        sys.exit(1)
    if sys.argv[1] == 'hydrogen':
        GLOBALS['m'] = 1.0
        GLOBALS['minv'] = 1.0
        GLOBALS['alpha'] = 1.0
        form = subprocess.run(["form", "hydrogen.frm"], capture_output=True).stdout
        sdp = make_sdp(form)
        print(solve(sdp, verbose=True))
    elif sys.argv[1] == 'dihydrogen':
        # TODO it is possible for there to be an ``emergent'' affine
        # constraint... how to deal with this?
        R = 1.0
        GLOBALS['m'] = 1.0
        GLOBALS['minv'] = 1.0
        GLOBALS['alpha'] = 1.0
        GLOBALS['R'] = R
        form = subprocess.run(["form", "dihydrogen.frm"], capture_output=True).stdout
        sdp = make_sdp(form)
        print(solve(sdp, verbose=True))
    else:
        print(f'Unknown problem: {sys.argv[1]}')
        sys.exit(1)

