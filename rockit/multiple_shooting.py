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

from .sampling_method import SamplingMethod
from casadi import sumsqr, vertcat, linspace, substitute, MX, evalf, vcat, horzsplit, veccat, DM, repmat
import numpy as np

class MultipleShooting(SamplingMethod):
    def __init__(self, *args, **kwargs):
        SamplingMethod.__init__(self, *args, **kwargs)

    def add_variables(self, stage, opti):
        self.add_time_variables(stage, opti)
        # We are creating variables in a special order such that the resulting constraint Jacobian
        # is block-sparse
        self.X.append(opti.variable(stage.nx))
        self.add_variables_V(stage, opti)

        for k in range(self.N):
            self.U.append(opti.variable(stage.nu))
            self.X.append(opti.variable(stage.nx))
            self.add_variables_V_control(stage, opti, k)

    def add_constraints(self, stage, opti):
        # Obtain the discretised system
        F = self.discrete_system(stage)

        if stage.is_free_time():
            opti.subject_to(self.T >= 0)

        self.xk = []
        self.q = 0
        # we only save polynomal coeffs for runge-kutta4
        if stage._method.intg == 'rk':
            self.poly_coeff = []
        else:
            self.poly_coeff = None

        for k in range(self.N):
            FF = F(x0=self.X[k], u=self.U[k], t0=self.control_grid[k],
                   T=self.control_grid[k + 1] - self.control_grid[k], p=vertcat(veccat(*self.P), self.get_p_control_at(stage, k)))
            # Dynamic constraints a.k.a. gap-closing constraints
            opti.subject_to(self.X[k + 1] == FF["xf"])

            # Save intermediate info
            poly_coeff_temp = FF["poly_coeff"]
            xk_temp = FF["Xi"]
            self.q = self.q + FF["qf"]
            # we cannot return a list from a casadi function
            self.xk.extend([xk_temp[:, i] for i in range(self.M)])
            if self.poly_coeff is not None:
                self.poly_coeff.extend(horzsplit(poly_coeff_temp, poly_coeff_temp.shape[1]//self.M))

            for l in range(self.M):
                for c, meta, _ in stage._constraints["integrator"]:
                    opti.subject_to(self.eval_at_integrator(stage, c, k, l), meta=meta)
                for c, meta, _ in stage._constraints["inf"]:
                    self.add_inf_constraints(stage, opti, c, k, l, meta)

            for c, meta, _ in stage._constraints["control"]:  # for each constraint expression
                opti.subject_to(self.eval_at_control(stage, c, k), meta=meta)

        for c, meta, _ in stage._constraints["control"]+stage._constraints["integrator"]:  # for each constraint expression
            # Add it to the optimizer, but first make x,u concrete.
            opti.subject_to(self.eval_at_control(stage, c, -1), meta=meta)

        self.xk.append(self.X[-1])

        for c, meta, _ in stage._constraints["point"]:  # Append boundary conditions to the end
            opti.subject_to(self.eval(stage, c), meta=meta)