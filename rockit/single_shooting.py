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
from casadi import sumsqr, horzcat, linspace, substitute, MX, evalf, vcat, horzsplit, veccat, DM, repmat, vvcat, vcat, vec
import numpy as np

class SingleShooting(SamplingMethod):
    def __init__(self, **kwargs):
        SamplingMethod.__init__(self, **kwargs)

    def add_parameter(self, stage, opti):
        SamplingMethod.add_parameter(self, stage, opti)
        self.Z0 = []
        es = []
        for s in stage.algebraics:
            e = opti.parameter(s.numel())
            opti.set_value(e, 0)
            es.append(e)
        self.Z0.append(vcat(es))
        self.Z0 += [None]*(self.N-1)

    def add_variables(self,stage,opti):
        scale_x = stage._scale_x
        scale_u = stage._scale_u

        # We are creating variables in a special order such that the resulting constraint Jacobian
        # is block-sparse
        self.X.append(vcat([opti.variable(s.numel(), scale=vec(stage._scale[s])) for s in stage.states]))
        self.add_variables_V(stage, opti)
        self.Q.append(DM.zeros(stage.nxq))

        for k in range(self.N):
            self.U.append(vcat([opti.variable(s.numel(), scale=vec(stage._scale[s])) for s in stage.controls]) if stage.nu>0 else MX(0,1))
            self.X.append(None)
            self.Q.append(None)
            self.add_variables_V_control(stage, opti, k)

        self.add_variables_V_control_finalize(stage, opti)

    def add_constraints(self,stage,opti):
        self.add_constraints_before(stage, opti)
        # Obtain the discretised system
        F = self.discrete_system(stage)

        # we only save polynomal coeffs for runge-kutta4
        if F.numel_out("poly_coeff")==0:
            self.poly_coeff = None
        if F.numel_out("poly_coeff_z")==0:
            self.poly_coeff_z = None
        if F.numel_out("poly_coeff_q")==0:
            self.poly_coeff_q = None

        Z0 = self.Z0[0]

        self.q = DM.zeros(stage.nxq)
        FFs = []
        self.xqk.append(DM.zeros(stage.nxq))
        # Fill in Z variables up-front, since they might be needed in constraints with ocp.next
        for k in range(self.N):
            FF = F(x0=self.X[k], u=self.U[k], t0=self.control_grid[k],
                   T=self.control_grid[k + 1] - self.control_grid[k], p=self.get_p_sys(stage, k), z0=Z0)
            self.X[k+1] = FF["xf"]
            poly_coeff_temp = FF["poly_coeff"]
            poly_coeff_z_temp = FF["poly_coeff_z"]
            poly_coeff_q_temp = FF["poly_coeff_q"]
            xk_temp = FF["Xi"]
            xqk_temp = self.q + FF["Qi"]
            zk_temp = FF["Zi"]
            # we cannot return a list from a casadi function
            self.xk.extend([xk_temp[:, i] for i in range(self.M)])
            self.xqk.extend([xqk_temp[:, i] for i in range(self.M)])
            self.zk.extend([zk_temp[:, i] for i in range(self.M)])
            if k==0:
                self.Z.append(zk_temp[:, 0])
            self.Z.append(FF["zf"])
            if self.poly_coeff is not None:
                self.poly_coeff.extend(horzsplit(poly_coeff_temp, poly_coeff_temp.shape[1]//self.M))
            if self.poly_coeff_q is not None:
                self.poly_coeff_q.extend(horzsplit(poly_coeff_q_temp, poly_coeff_q_temp.shape[1]//self.M))
            if self.poly_coeff_z is not None:
                self.poly_coeff_z.extend(horzsplit(poly_coeff_z_temp, poly_coeff_z_temp.shape[1]//self.M))
            FFs.append(FF)
            self.q = self.q + FF["qf"]
            self.Q[k+1] = self.q

        self.xk.append(self.X[-1])
        self.zk.append(self.zk[-1])

        for k in range(self.N):
            FF = FFs[k]

            for l in range(self.M):
                for c, meta, args in stage._constraints["integrator"]:
                    if k==0 and l==0 and not args["include_first"]: continue
                    opti.subject_to(self.eval_at_integrator(stage, c, k, l), scale=args["scale"], meta=meta)
                for c, meta, _ in stage._constraints["inf"]:
                    self.add_inf_constraints(stage, opti, c, k, l, meta)

            for c, meta, args in stage._constraints["control"]:  # for each constraint expression
                if k==0 and not args["include_first"]: continue
                try:
                    opti.subject_to(self.eval_at_control(stage, c, k), scale=args["scale"], meta=meta)
                except IndexError:
                    pass # Can be caused by ocp.offset -> drop constraint

        for c, meta, args in stage._constraints["control"]+stage._constraints["integrator"]:  # for each constraint expression
            if not args["include_last"]: continue
            # Add it to the optimizer, but first make x,u concrete.
            try:
                opti.subject_to(self.eval_at_control(stage, c, -1), scale=args["scale"], meta=meta)
            except IndexError:
                pass # Can be caused by ocp.offset -> drop constraint
