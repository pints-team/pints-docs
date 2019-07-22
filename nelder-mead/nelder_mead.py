#/usr/bin/env python3
from __future__ import division
import numpy as np
import pints


class NelderMead(pints.Optimiser):
    """
    Nelder-Mead downhill simplex method.

    Note: this is a deterministic local optimiser, and will not typically
    benefit from parallelisation.

    Implementation of the classical algorithm by [1], following the
    presentation in Algorithm 8.1 of [2].

    Generates a "simplex" of ``n + 1`` samples around a given starting point,
    and evaluates their scores. Next, each iteration consists of a sequence of
    operations, typically the worst sample ``y_worst`` is replaced with a new
    point::

        y_new = mu + delta * (mu - y_worst)
        mu = (1 / n) * sum(y), y != y_worst

    where ``delta`` has one of four values, depending on the type of operation:

    - Reflection (``delta = 1``)
    - Expansion (``delta = 2``)
    - Inside contraction (``delta = -0.5``)
    - Outside contraction (``delta = 0.5``)

    Note that the ``delta`` values here are common choices, but not the only
    valid choices.

    A fifth type of iteration called a "shrink" is occasionally performed, in
    which all samples except the best sample ``y_best`` are replaced::

        y_i_new = y_best + ys * (y_i - y_best)

    where ys is a parameter (typically ys = 0.5).

    The initialisation of the initial simplex was copied from [3].

    [1] A simplex method for function minimization
    Nelder, Mead 1965, Computer Journal
    https://doi.org/10.1093/comjnl/7.4.308

    [2] Introduction to derivative-free optimization
    Andrew R. Conn, Katya Scheinberg, Luis N. Vicente
    2009, First edition. ISBN 978-0-098716-68-9
    https://doi.org/10.1137/1.9780898718768

    [3] SciPy
    https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py#L455
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(NelderMead, self).__init__(x0, sigma0, boundaries)

        # Parameters for reflection, expansion, and contraction
        # Other choices can be made, as long as -1 < di < 0 < do < dr < de
        self._di = -0.5   # Inside contraction
        self._do = 0.5    # Outside contraction
        self._dr = 1      # Reflection
        self._de = 2      # Expansion

        # Parameter for shrinking
        # Other choices are ok, as long as 0 < ys < 1
        self._ys = 0.5

        # Points and scores
        self._xs = None
        self._fs = None

        # Status
        self._running = False
        self._xm = None             # Mean of n best
        self._xr = self._fr = None  # Reflected point and score
        self._xp = None             # Proposed expansion/contraction point
        self._shrink = False        # Shrink step required



        self.ax = None
        self.last = x0



    def ask(self):
        """ See: :meth:`pints.Optimiser.ask()`. """

        # Initialise
        if self._xs is None:
            self._running = True

            # Initialise simplex, reticulate splines
            # TODO: Use sigma here?
            n = self._n_parameters
            self._xs = np.zeros((n + 1, n))
            self._xs[0] = self._x0
            x_grow = 1.05       # From scipy implementation
            x_zero = 0.00025    # From scipy implementation
            for i in range(n):
                self._xs[1 + i] = self._x0
                if self._xs[1 + i][i] == 0:
                    self._xs[1 + i][i] = x_zero
                else:
                    self._xs[1 + i][i] *= x_grow

            # Ask for initial points
            return np.array(self._xs, copy=True)

        # Shrink operation
        if self._shrink:
            for i in range(self._n_parameters):
                self._xs[1 + i] = \
                    self._xs[0] + self._ys * (self._xs[1 + i] - self._xs[0])

            return np.array(self._xs[1:], copy=True)

        # Start of normal iteration, ask for reflection point
        if self._xr is None:

            #TODO
            self._xm = np.mean(self._xs[:-1], axis=0)

            # Order, and calculate mean of all except worst
            ifs = np.argsort(self._fs)
            self._xs = self._xs[ifs]
            self._fs = self._fs[ifs]
            self._xm = np.mean(self._xs[:-1], axis=0)

            # TODO
            xxx = [self.last[0], self._xs[0][0]]
            yyy = [self.last[1], self._xs[0][1]]
            self.ax.plot(xxx, yyy, 'x-', markersize=20)
            self.last = self._xs[0]

            # Calculate reflection point and return
            self._xr = self._xm + self._dr * (self._xm - self._xs[-1])

            # Ask for f(xr)
            return np.array([self._xr])

        # Extended iteration: ask for expansion or contraction point
        return np.array([self._xp])


    def fbest(self):
        """ See: :meth:`pints.Optimiser.fbest()`. """
        if not self._running:
            raise RuntimeError('Best score cannot be returned before run.')
        return self._fs[0]

    def name(self):
        """ See: :meth:`pints.Optimiser.name()`. """
        return 'Nelder-Mead simplex'

    def running(self):
        """ See: :meth:`pints.Optimiser.running()`. """
        return self._running

    def stop(self):
        """ See: :meth:`pints.Optimiser.stop()`. """
        return False

    def tell(self, fx):
        """ See: :meth:`pints.Optimiser.tell()`. """

        # Initialise
        if self._fs is None:
            fx = np.array(fx, copy=True)
            if np.prod(fx.shape) != self._n_parameters + 1:
                raise ValueError(
                    'Expecting a vector of length (1 + n_parameters).')
            self._fs = fx.reshape((1 + self._n_parameters, ))

            # Set out-of-bounds to inf
            if self._boundaries is not None:
                for i, x in enumerate(self._xs[1:]):
                    if not self._boundaries.check(x):
                        self._fs[1 + i] = np.inf

            # Return: ask will set xr, get f(xr)
            return

        # Shrink
        if self._shrink:
            fx = np.array(fx, copy=False)
            if np.prod(fx.shape) != self._n_parameters:
                raise VaueError(
                    'Expecting a vector of length n_parameters.')
            self._fs[1:] = fx

            # Set out-of-bounds to inf
            if self._boundaries is not None:
                for i, x in enumerate(self._xs[1:]):
                    if not self._boundaries.check(x):
                        self._fs[1 + i] = np.inf

            # Reset and return: ask will set xr, get f(xr)
            self._shrink = False
            self._xp = self._xr = self._fr = None
            return

        # Reflection point or contraction/expansion point returned
        if len(fx) != 1:
            raise ValueError('Expecting only a single evaluation')
        fx = fx[0]

        # Set out-of-bounds fs to inf
        if self._boundaries is not None:
            x = self._xr if self._fr is None else self._xp
            if not self._boundaries.check(x):
                fx = np.inf

        # Set reflection point
        if self._fr is None:
            self._fr = fx

        # Determine operation
        expand = self._fr < self._fs[0]
        contract = self._fr >= self._fs[-2]

        # Option 1: Reflection
        if not (expand or contract):
            self._xs[-1] = self._xr
            self._fs[-1] = self._fr

            # Reset and return: ask will set xr, get f(xr)
            self._xr = self._fr = None
            return

        # Option 2: Expansion
        if expand:
            # Propose expansion
            if self._xp is None:
                self._xp = self._xm + self._de * (self._xm - self._xs[-1])
                return

            # Accept or reject expansion
            if fx <= self._fr:
                self._xs[-1] = self._xp
                self._fs[-1] = fx
            else:
                self._xs[-1] = self._xr
                self._fs[-1] = self._fr

            # Reset and return: ask will set xr, get f(xr)
            self._xp = self._xr = self._fr = None
            return

        # Option 3/4: Outside/Inside contraction
        outside = self._fr < self._fs[-1]

        # Propose contraction
        if self._xp is None:
            dc = self._do if outside else self._di
            self._xp = self._xm + dc * (self._xm - self._xs[-1])
            return

        # Accept contraction, or shrink
        if ((fx <= self._fr) if outside else (fx < self._fs[-1])):
            self._xs[-1] = self._xp
            self._fs[-1] = fx

            # Reset and return: ask will set xr, get f(xr)
            self._xp = self._xr = self._fr = None
            return

        # Option 5: Shrink
        self._shrink = True

    def xbest(self):
        """ See: :meth:`pints.Optimiser.xbest()`. """
        if not self._running:
            raise RuntimeError('Best position cannot be returned before run.')
        return pints.vector(self._xs[0])


class CircularBoundaries(pints.Boundaries):
    """
    Circular boundaries, to test boundaries that are non-rectangular.

    Arguments:

    ``center``
        The point these boundaries are centered on.
    ``radius``
        The radius (in all directions).

    """
    def __init__(self, center, radius=1):
        super(CircularBoundaries, self).__init__()

        # Check arguments
        center = pints.vector(center)
        if len(center) < 1:
            raise ValueError('Number of parameters must be at least 1.')
        self._center = center
        self._n_parameters = len(center)

        radius = float(radius)
        if radius <= 0:
            raise ValueError('Radius must be greater than zero.')
        self._radius2 = radius**2

    def check(self, parameters):
        """ See :meth:`pints.Boundaries.check()`. """
        return np.sum((parameters - self._center)**2) < self._radius2

    def n_parameters(self):
        """ See :meth:`pints.Boundaries.n_parameters()`. """
        return self._n_parameters

