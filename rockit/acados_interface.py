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

from .multiple_shooting import MultipleShooting
from .sampling_method import SamplingMethod
from .solution import OcpSolution
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from casadi import MX, vcat, symvar, substitute, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, mtimes, pinv, evalf
import casadi
import numpy as np
import scipy


class AcadosInterface:
    def __init__(self,N=20,**args):
        self.N = N
        self.args = args

    def inherit(self, parent):
        pass

    def fill_placeholders_t0(self, stage, expr, *args):
        return stage._t0

    def fill_placeholders_T(self, stage, expr, *args):
        return stage._T

    def transcribe_placeholders(self, stage, placeholders):
        """
        Transcription is the process of going from a continuous-time OCP to an NLP
        """
        return stage._transcribe_placeholders(self, placeholders)
 
    def fill_placeholders_integral_control(self, stage, expr, *args):
        return expr

    def fill_placeholders_at_t0(self, stage, expr, *args):
        return expr

    def fill_placeholders_at_tf(self, stage, expr, *args):
        return expr

    def main_transcribe(self, stage, pass_nr=1, **kwargs):
        pass

    def transcribe(self, stage, pass_nr=1, **kwargs):

        if pass_nr==1:
            self.stage = stage
            self.ocp = ocp = AcadosOcp()
            self.opti = Opti()

            self.X_gist = [MX.sym("X", stage.nx) for k in range(self.N+1)]
            self.U_gist = [MX.sym("U", stage.nu) for k in range(self.N)]

            model = AcadosModel()

            f = stage._ode()
            x = stage.x
            xdot = MX.sym("xdot", stage.nx)
            u = stage.u
            p = MX.sym("p", stage.np+stage.v.shape[0])

            res = f(x=x, u=u, p=p)
            f_expl = res["ode"]

            model.f_impl_expr = xdot-f_expl
            model.f_expl_expr = f_expl
            model.x = x
            model.xdot = xdot
            model.u = u
            model.p = p

            self.X = self.opti.variable(*x.shape)
            self.U = self.opti.variable(*u.shape)
            self.P = self.opti.parameter(*p.shape)

            self.raw = [x,u,p]
            self.optivar = [self.X, self.U, self.P]

            model.name = "rockit_model"

            ocp.model = model

            # set dimensions
            ocp.dims.N = self.N

            var_lagrange = []
            var_mayer = []
            for e in symvar(stage._objective):
                if "integral" in e.name():
                    var_lagrange.append(e)
                elif "at_tf" in e.name():
                    var_mayer.append(e)
                else:
                    raise Exception("Unknown element in objective: %s" % str(e))

            self.lagrange = substitute([stage._objective], var_mayer, [DM.zeros(e.shape) for e in var_mayer])[0]
            self.mayer = substitute([stage._objective], var_lagrange, [DM.zeros(e.shape) for e in var_lagrange])[0]

            self.initial_conditions = []
            self.final_conditions = []
        else:
            opti_advanced = self.opti.advanced
            placeholders = kwargs["placeholders"]
            lagrange = placeholders(self.lagrange)
            mayer = placeholders(self.mayer)

            ocp = self.ocp

            ocp.cost.cost_type = 'EXTERNAL'
            ocp.cost.cost_type_e = 'EXTERNAL'

            ocp.model.cost_expr_ext_cost = lagrange
            ocp.model.cost_expr_ext_cost_e = mayer

            Jbx = []; lbx = []; ubx = []
            Jbu = []; lbu = []; ubu = []
            C = []; D = []; lg = []; ug = []
            h = []; lbh = []; uh = []

            h_e = []; lbh_e = []; uh_e = []
            Jbx_e = []; lbx_e = []; ubx_e = []
            C_e = []; lg_e = []; ug_e = []

            x0_J = []
            x0_c = []


            def legit_J(J):
                try:
                    J = evalf(J)
                except:
                    return False
                # Check if slice of permutation of unit matrix
                if not np.all(np.array(J.colind())==np.array(range(J.shape[1]+1))):
                    return False
                if not np.all(np.array(J.nonzeros())==1):
                    return False
                return True

            for c, meta, args in stage._constraints["control"]+stage._constraints["integrator"]:
                c = substitute([placeholders(c)],self.raw,self.optivar)[0]
                mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
                lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)

                if is_linear(canon, vertcat(self.ocp.model.x,self.ocp.model.u)):
                    if not depends_on(canon, self.ocp.model.x):
                        # lbu <= Jbu u <= ubu
                        J, c = linear_coeff(canon, self.ocp.model.u)
                        if legit_J(J):
                            Jbu.append(J)
                            lbu.append(lb-c)
                            ubu.append(ub-c)
                            continue
                    if not depends_on(canon, self.ocp.model.u):
                        # lbx <= Jbx x <= ubx
                        J,c = linear_coeff(canon, self.ocp.model.x)
                        if legit_J(J):
                            Jbx.append(J)
                            lbx.append(lb-c)
                            ubx.append(ub-c)
                            continue

                    # lg <= Cx + Du <= ug
                    J,c = linear_coeff(canon, vertcat(self.ocp.model.x,self.ocp.model.u))
                    C.append(J[:,:stage.nx])
                    D.append(J[:,stage.nx:])
                    lg.append(lb-c)
                    ug.append(ub-c)
                else:
                    # lbh <= h(x,u) <= uh
                    h.append(canon)
                    lbh.append(lb)
                    uh.append(ub)

            for c, meta, _ in stage._constraints["point"]:
                has_t0 = 'r_at_t0' in [a.name() for a in symvar(c)]
                has_tf = 'r_at_tf' in [a.name() for a in symvar(c)]

                c = substitute([placeholders(c)],self.raw,self.optivar)[0]
                mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
                lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)

                if has_t0 and has_tf:
                    raise Exception("Cosntraints mixing value at t0 and tf not allowed.")

                if has_t0:
                    # t0
                    check = mc.type == casadi.OPTI_EQUALITY
                    check = check and is_linear(canon, self.ocp.model.x)
                    check = check and not depends_on(canon, self.ocp.model.u)
                    assert check, "at t=t0, only equality constraints on x are allowed. Got '%s'" % str(c)

                    J,c = linear_coeff(canon, self.ocp.model.x)
                    x0_J.append(J)
                    x0_c.append(lb-c)
                else:
                    # tf
                    assert not depends_on(canon,ocp.model.u), "Terminal constraints cannot depend on u"
                    if is_linear(canon, self.ocp.model.x):
                        # lbx <= Jbx x <= ubx
                        J,c = linear_coeff(canon, self.ocp.model.x)
                        if legit_J(J):
                            Jbx_e.append(J)
                            lbx_e.append(lb-c)
                            ubx_e.append(ub-c)
                            continue
                        C_e.append(J)
                        lg_e.append(lb-c)
                        ug_e.append(lb-c)
                    else:
                        # lbh <= h(x,u) <= uh
                        h_e.append(canon)
                        lbh_e.append(lb)
                        uh_e.append(ub)

            x0 = mtimes(pinv(vcat(x0_J)), vcat(x0_c))

            # Parametric not supported yet
        

            def export_expr(m):
                if isinstance(m,list):
                    if len(m)==0:
                        return MX(0, 1)
                    else:
                        return vcat(m)
                return m

            def export(m):
                print(np.array(evalf(export_expr(m))))
                return np.array(evalf(export_expr(m)))

            def export_vec(m):
                return export(m).reshape((-1))

            ocp.constraints.constr_type = 'BGH'

            if Jbx:
                ocp.constraints.Jbx = export(Jbx)
                ocp.constraints.lbx = export_vec(lbx)
                ocp.constraints.ubx = export_vec(ubx)

            if Jbu:
                ocp.constraints.Jbu = export(Jbu)
                ocp.constraints.lbu = export_vec(lbu)
                ocp.constraints.ubu = export_vec(ubu)

            if C:
                ocp.constraints.C = export(C)
                ocp.constraints.D = export(D)
                ocp.constraints.lg = export_vec(lg)
                ocp.constraints.ug = export_vec(ug)

            if h:
                ocp.model.con_h_expr = export_expr(h)
                ocp.constraints.lh = export_vec(lbh)
                ocp.constraints.uh = export_vec(uh)

            if h_e:
                ocp.model.con_h_e_expr = export_expr(h_e)
                ocp.constraints.lbh = export_vec(lbh_e)
                ocp.constraints.uh = export_vec(uh_e)

            if Jbx_e:
                ocp.constraints.Jbx_e = export(Jbx_e)
                ocp.constraints.lbx_e = export_vec(lbx_e)
                ocp.constraints.ubx_e = export_vec(ubx_e)

            if C_e:
                ocp.constraints.C_e = export(C_e)
                ocp.constraints.lg_e = export_vec(lg_e)
                ocp.constraints.ug_e = export_vec(ug_e)

            ocp.constraints.x0 = export_vec(x0)


            for k, v in self.args.items():
                setattr(ocp.solver_options, k, v)

            ocp.solver_options.tf = stage._T
            self.control_grid = casadi.linspace(MX(stage._t0), stage._t0+stage._T, self.N+1)

            self.ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + ocp.model.name + '.json')

    def solve(self, stage):
        status = self.ocp_solver.solve()
        self.ocp_solver.print_statistics()
        x = [self.ocp_solver.get(i, "x") for i in range(self.N+1)]
        u = [self.ocp_solver.get(i, "u") for i in range(self.N)]
        print("x", x)
        print("u", u)
        return OcpSolution(SolWrapper(self, vcat(x), vcat(u)), stage)

    @property
    def gist(self):
        """Obtain an expression packing all information needed to obtain value/sample

        The composition of this array may vary between rockit versions

        Returns
        -------
        :obj:`~casadi.MX` column vector

        """
        return vertcat(vcat(self.X_gist), vcat(self.U_gist))

    def eval_at_control(self, stage, expr, k):
        return substitute([expr],[self.ocp.model.x,self.ocp.model.u],[self.X_gist[k], self.U_gist[min(k, self.N-1)]])[0]

class SolWrapper:
    def __init__(self, method, x, u):
        self.method = method
        self.x = x
        self.u = u

    def value(self, expr, *args,**kwargs):
        placeholders = self.method.stage.placeholders_transcribed
        return np.array(evalf(substitute([placeholders(expr)],[self.method.gist],[vertcat(self.x, self.u)])[0]))