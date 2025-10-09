#!/usr/bin/env python

import ast
import re
import sys

import numpy as np

i_ = 1j

def parse(s):
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'-', '+-', s)
    #s = re.sub(r'oprod\(([^)]*)\)', r'oprod("\1")', s)
    #s = re.sub(r'\^', '**', s)
    terms = dict()
    for term in s.split('+'):
        op = ''
        c = 0
        coef = 1.0
        for fact in term.split('*'):
            if fact[0] == '-' and fact[1:6] == 'oprod':
                fact = fact[1:]
                coef *= -1
            if fact[0] == '-' and fact[1] == 'c':
                fact = fact[1:]
                coef *= -1
            if m := re.match(r'oprod\(([^)]+)\)', fact):
                op = m[1]
            elif m := re.match(r'c\(([^)]+)\)', fact):
                c = int(m[1])
            else:
                coef *= eval(fact)
            terms[op,c] = coef
    return terms

if __name__ == '__main__':
    with open('hamiltonian.expr') as f:
        hamiltonian = parse(f.read())
    with open('sos.expr') as f:
        sos = parse(f.read())

    opset = set()
    cset = set()
    for op,c in hamiltonian:
        opset.add(op)
        cset.add(c)
    for op,c in sos:
        opset.add(op)
        cset.add(c)

    ops = list(opset)
    cs = list(cset)

    print(hamiltonian)

