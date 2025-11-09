#!/usr/bin/env python

import jax
import jax.numpy as jnp

import numpy as np
import numpy.random as nr

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
        while True:
            break
        # TODO M0
        #return SemidefiniteProgram(M0, m, c)

class InteriorPointSolver:
    def __init__(self, sdp):
        pass

    def solve(self):
        pass

