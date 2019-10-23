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
from numpy import nan

class OcpSolution:
    def __init__(self, nlpsol, stage):
        """Wrap casadi.nlpsol to simplify access to numerical solution."""
        self.sol = nlpsol
        self.stage = stage

    def __call__(self, stage):
        """Sample expression at solution on a given grid.

        Parameters
        ----------
        stage : :obj:`~rockit.stage.Stage`
            An optimal control problem stage.
        """
        return OcpSolution(self.sol, stage=stage)

    def value(self, expr, *args, **kwargs):
        """Get the value of an (non-signal) expression.

        Parameters
        ----------
        expr : :obj:`casadi.MX`
            Arbitrary expression containing no signals (states, controls) ...
        """
        return self.sol.value(self.stage.value(expr, *args, **kwargs))

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

        expr_shape = MX(expr).shape
        expr_prod = expr_shape[0]*expr_shape[1]

        tdim = time.numel()
        target_shape = (tdim,)+tuple([e for e in expr_shape if e!=1])

        res = self.sol.value(res).reshape(expr_shape[0], tdim, expr_shape[1])
        res = np.transpose(res,[1,0,2])
        res = res.reshape(target_shape)
        return self.sol.value(time), res