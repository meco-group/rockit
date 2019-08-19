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
from casadi import vertcat, vcat, DM, Function, hcat
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
        return self.sol.value(expr, *args, **kwargs)

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
        if grid == 'control':
            return self._grid_control(self.stage, expr, grid, **kwargs)
        elif grid == 'integrator':
            if 'refine' in kwargs:
                return self._grid_intg_fine(self.stage, expr, grid, **kwargs)
            else:
                return self._grid_integrator(self.stage, expr, grid, **kwargs)
        elif grid == 'integrator_roots':
            return self._grid_integrator_roots(self.stage, expr, grid, **kwargs)
        else:
            msg = "Unknown grid option: {}\n".format(grid)
            msg += "Options are: 'control' or 'integrator' with an optional extra refine=<int> argument."
            raise Exception(msg)

    def _grid_control(self, stage, expr, grid):
        """Evaluate expression at (N + 1) control points."""
        sub_expr = []
        for k in list(range(stage._method.N))+[-1]:
            sub_expr.append(stage._method.eval_at_control(stage, expr, k))
        res = [self.sol.value(elem) for elem in sub_expr]
        time = self.sol.value(stage._method.control_grid)
        return time, np.array(res)

    def _grid_integrator(self, stage, expr, grid):
        """Evaluate expression at (N*M + 1) integrator discretization points."""
        sub_expr = []
        time = []
        for k in range(stage._method.N):
            for l in range(stage._method.M):
                sub_expr.append(stage._method.eval_at_integrator(stage, expr, k, l))
            time.append(stage._method.integrator_grid[k])
        sub_expr.append(stage._method.eval_at_control(stage, expr, -1))
        res = [self.sol.value(elem) for elem in sub_expr]
        time = self.sol.value(vcat(time))
        return time, np.array(res)

    def _grid_integrator_roots(self, stage, expr, grid):
        """Evaluate expression at integrator roots."""
        sub_expr = []
        tr = []
        for k in range(stage._method.N):
            for l in range(stage._method.M):
                for j in range(stage._method.xr[k][l].shape[1]):
                    sub_expr.append(stage._method.eval_at_integrator_root(stage, expr, k, l, j))
                tr.extend(stage._method.tr[k][l])
        res = [self.sol.value(elem) for elem in sub_expr]
        time = self.sol.value(hcat(tr))
        return time, np.array(res)

    def _grid_intg_fine(self, stage, expr, grid, refine):
        """Evaluate expression at extra fine integrator discretization points."""
        if stage._method.poly_coeff is None:
            msg = "No polynomal coefficients for the {} integration method".format(stage._method.intg)
            raise Exception(msg)
        N, M, T = stage._method.N, stage._method.M, stage._method.T

        expr_f = Function('expr', [stage.x, stage.z, stage.u], [expr])

        sub_expr = []

        time = self.sol.value(stage._method.control_grid)
        total_time = []
        for k in range(N):
            t0 = time[k]
            dt = (time[k+1]-time[k])/M
            tlocal = np.linspace(0, dt, refine + 1) 
            ts = DM(tlocal[:-1]).T
            for l in range(M):
                total_time.append(t0+tlocal[:-1])
                coeff = stage._method.poly_coeff[k * M + l]
                tpower = vcat([ts**i for i in range(coeff.shape[1])])
                if stage._method.poly_coeff_z:
                    coeff_z = stage._method.poly_coeff_z[k * M + l]
                    tpower_z = vcat([ts**i for i in range(coeff_z.shape[1])])
                    z = coeff_z @ tpower_z
                else:
                    z = nan
                sub_expr.append(expr_f(coeff @ tpower, z, stage._method.U[k]))
                t0+=dt

        ts = tlocal[-1]
        total_time.append(time[k+1])
        tpower = vcat([ts**i for i in range(coeff.shape[1])])
        if stage._method.poly_coeff_z:
            tpower_z = vcat([ts**i for i in range(coeff_z.shape[1])])
            z = coeff_z @ tpower_z
        else:
            z = nan
        sub_expr.append(expr_f(stage._method.poly_coeff[-1] @ tpower, z, stage._method.U[-1]))

        res = self.sol.value(hcat(sub_expr))

        time = np.hstack(total_time)
        return time, res
