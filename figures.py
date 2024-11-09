#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

with open('x') as f:
    dat = np.array([[float(x) for x in l.split()] for l in f.readlines()])

plt.figure(figsize=(5,3), dpi=300)

# Plot exact
cond = dat[:,1] == -1
d = dat[cond,:]
plt.plot(d[:,0], d[:,3], color='#000000')

# Plot (4,0) bound
cond = (dat[:,1] == 4) * (dat[:,2] == 0)
d = dat[cond,:]
plt.fill_between(d[:,0], d[:,3], d[:,4], color='#000000', alpha=0.1)

# Plot (4,3) bound
#cond = (dat[:,1] == 4) * (dat[:,2] == 3)
#d = dat[cond,:]
#plt.fill_between(d[:,0], d[:,3], d[:,4], color='#550000', alpha=0.2)

# Plot (9,0) bound
#cond = (dat[:,1] == 9) * (dat[:,2] == 0)
#d = dat[cond,:]
#plt.fill_between(d[:,0], d[:,3], d[:,4], color='#000066', alpha=0.3)

# Plot (9,1) bound
cond = (dat[:,1] == 9) * (dat[:,2] == 1)
d = dat[cond,:]
plt.fill_between(d[:,0], d[:,3], d[:,4], color='#008800', alpha=0.4)

# Plot (9,2) bound
cond = (dat[:,1] == 9) * (dat[:,2] == 2)
d = dat[cond,:]
plt.fill_between(d[:,0], d[:,3], d[:,4], color='#000000', alpha=0.5)

plt.xlabel('$T$')
plt.ylabel('$\\langle x(T)\\rangle$')
#plt.ylim([-0.08, 0.08])
plt.ylim([-1.0, 1.0])
plt.xlim([0,2])
plt.tight_layout()
plt.savefig("aho.png")

