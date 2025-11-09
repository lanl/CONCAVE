#!/usr/bin/env python

import re
import subprocess
import sys

from sdp import *

GLOBALS = {}

class Term:
    def __init__(self, coef, num=1, op='I'):
        self.num = num
        self.coef = coef
        self.op = op

    def __repr__(self):
        return f"{self.num}*{self.coef}*[{self.op}]"

    @staticmethod
    def parse(s):
        ops = 'I'
        coefl = (None,None)
        coefr = (None,None)
        c = 1
        match = re.match(r'(?ms)([+-])\s+(.*)',s)
        if match.group(1) == '-':
            c *= -1
        for factor in match.group(2).split('*'):
            if m := re.match(r'oprod\(([^)]+)\)', factor):
                ops = m.group(1)
            elif m := re.match(r'conj\(c\(([0-9]+),([0-9]+)\)\)', factor):
                if coefl != (None,None):
                    raise Exception("Two left coefficients")
                coefl = (int(m.group(1)),int(m.group(2)))
            elif m := re.match(r'c\(([0-9]+),([0-9]+)\)', factor):
                if coefr != (None,None):
                    raise Exception("Two right coefficients")
                coefr = (int(m.group(1)),int(m.group(2)))
            elif re.match(r'i_', factor):
                c *= 1j
            else:
                n = eval(factor, GLOBALS)
                if type(n) == float:
                    c *= n
                else:
                    raise Exception("What's this: "+factor)
        return Term((coefl,coefr), c, ops)

def parse_expression(expr):
    # Split into terms. A term is a thing that begins ^\s+[+-].
    term_strings = re.findall(r'(?ms)\s+([+-].+?)(?=\s+(?:[+-]|\Z))' ,expr)
    terms = list(map(Term.parse, term_strings))
    return terms

def make_sdp(form):
    if type(form) == bytes:
        form = form.decode()
    # Extract the two expressions.
    ham_start = form.index('hamiltonian =') + len('hamiltonian =')
    ham_end = form.index(';', ham_start)
    ham = parse_expression(form[ham_start:ham_end])
    sos_start = form.index('sos =') + len('sos =')
    sos_end = form.index(';', sos_start)
    sos = parse_expression(form[sos_start:sos_end])

    print(ham)
    print(sos)

def solve(sdp):
    ipm = InteriorPointSolver()

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
        print(solve(sdp))
    elif sys.argv[1] == 'dihydrogen':
        form = subprocess.run(["form", "dihydrogen.frm"], capture_output=True).stdout
        sdp = make_sdp(form)
        print(solve(sdp))
    else:
        print('Unknown problem')
        sys.exit(1)

