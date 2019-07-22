#!/usr/bin/env python3
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pints
import pints.toy
import pints.plot

from nelder_mead import NelderMead, CircularBoundaries

# Create Rosenbrock error
f = pints.toy.RosenbrockError()

# Choose starting position
x0 = [-0.75, 3.5]

# Create figure
fig = plt.figure(figsize=(9, 9))
fig.subplots_adjust(0.05, 0.05, 0.98, 0.98)

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Show function
if True:
    x = np.linspace(-1.5, 1.5, 30)
    y = np.linspace(-0.5, 4, 45)
    X, Y = np.meshgrid(x, y)
    Z = [[np.log(f([i, j])) for i in x] for j in y]
    levels = np.linspace(np.min(Z), np.max(Z), 20)
    ax.contour(X, Y, Z, levels = levels)

# Run scipy optimisation
x1_path = [x0]

def update_path(x, res=None):
    x1_path.append(np.copy(x))


# Run own version with boundaries
c = 1, 1
r = 3.06
ax.add_artist(plt.Circle(c, r, color='green', alpha=0.1))
b = CircularBoundaries(c, r)

opt = pints.OptimisationController(f, x0, boundaries=b, method=NelderMead)
opt._optimiser.ax = ax
x, _ = opt.run()


plt.show()

