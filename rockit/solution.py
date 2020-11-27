#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

import numpy as np
from casadi import vertcat, vcat, DM, Function, hcat, MX
from .casadi_helpers import DM2numpy
from numpy import nan
import functools

class OcpSolution:
    def __init__(self, nlpsol, stage):
        """Wrap casadi.nlpsol to simplify access to numerical solution."""
        self.sol = nlpsol
        self.stage = stage._augmented # SHould this be ocP?
        self._gist = np.array(nlpsol.value(self.stage.master.gist)).squeeze()

    def __call__(self, stage):
        """Sample expression at solution on a given grid.

        Parameters
        ----------
        stage : :obj:`~rockit.stage.Stage`
            An optimal control problem stage.
        """
        return OcpSolution(self.sol, stage=stage)

    def value(self, expr, *args):
        """Get the value of an (non-signal) expression.

        Parameters
        ----------
        expr : :obj:`casadi.MX`
            Arbitrary expression containing no signals (states, controls) ...
        """
        return self.sol.value(self.stage.value(expr, *args))

    def sample(self, expr, grid, **kwargs):
        """Sample expression at solution on a given grid.

        Parameters
        ----------
        expr : :obj:`casadi.MX`
            Arbitrary expression containing states, controls, ...
        grid : `str`
            At which points in time to sample, options are
            'control' or 'integrator' (at integrator discretization
            level) or 'integrator_roots'.
        refine : int, optional
            Refine grid by evaluation the polynomal of the integrater at
            intermediate points ("refine" points per interval).

        Returns
        -------
        time : numpy.ndarray
            Time from zero to final time, same length as res
        res : numpy.ndarray
            Numerical values of evaluated expression at points in time vector.

        Examples
        --------
        Assume an ocp with a stage is already defined.

        >>> sol = ocp.solve()
        >>> tx, xs = sol.sample(x, grid='control')
        """
        time, res = self.stage.sample(expr, grid, **kwargs)
        res = self.sol.value(res)
        return self.sol.value(time), DM2numpy(res, MX(expr).shape, time.numel())

    def sampler(self, *args):
        """Returns a function that samples given expressions


        This function has two modes of usage:
        1)  sampler(exprs)  -> Python function
        2)  sampler(name, exprs, options) -> CasADi function

        Parameters
        ----------
        exprs : :obj:`casadi.MX` or list of :obj:`casadi.MX`
            List of arbitrary expression containing states, controls, ...
        name : `str`
            Name for CasADi Function
        options : dict, optional
            Options for CasADi Function

        Returns
        -------
        t -> output

        mode 1 : Python Function
            Symbolically evaluated expression at points in time vector.
        mode 2 : :obj:`casadi.Function`
            Time from zero to final time, same length as res

        Examples
        --------
        Assume an ocp with a stage is already defined.

        >>> sol = ocp.solve()
        >>> s = sol.sampler(x)
        >>> s(1.0) # Value of x at t=1.0
        """
        s = self.stage.sampler(*args)
        ret = functools.partial(s, self.gist)
        ret.__doc__ = """
                Parameters
                ----------
                t : float or float vector
                    time or time-points to sample at

                Returns
                -------
                :obj:`np.array`

                """
        return ret

    @property
    def gist(self):
        """All numerical information needed to compute any value/sample

        Returns
        -------
        1D numpy.ndarray
           The composition of this array may vary between rockit versions


        """
        return self._gist

    @property
    def stats(self):
        """Retrieve solver statistics

        Returns
        -------
        Dictionary
           The information contained is not structured and may change between rockit versions

        """
        return self.sol.stats()