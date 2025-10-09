#!/usr/bin/env python

import subprocess
import sys

import sdp

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: ./solve.py (hydrogen|dihydrogen)')
        sys.exit(1)
    if sys.argv[1] == 'hydrogen':
        pass
    elif sys.argv[1] == 'dihydrogen':
        pass
    else:
        print('Unknown problem')
        sys.exit(1)

