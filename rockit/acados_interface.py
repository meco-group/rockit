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
from collections import OrderedDict
from casadi import Sparsity, MX, vcat, veccat, symvar, substitute, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag
import casadi
import numpy as np
import scipy

INF = 1e15


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

    def main_transcribe(self, stage, phase=1, **kwargs):
        pass

    def transcribe(self, stage, phase=1, **kwargs):

        if phase==1:
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
            p = stage.p

            res = f(x=x, u=u, p=p)
            f_expl = res["ode"]

            model.f_impl_expr = xdot-f_expl
            model.f_expl_expr = f_expl
            model.x = x
            model.xdot = xdot
            model.u = u
            model.p = p
            self.P0 = DM.zeros(stage.np)

            slack = MX(0, 1) if len(stage.variables['control'])==0 else vvcat(stage.variables['control'])
            self.slack = slack

            self.X = self.opti.variable(*x.shape)
            self.U = self.opti.variable(*u.shape)
            self.P = self.opti.parameter(*p.shape)
            self.V = self.opti.variable(*slack.shape)

            self.raw = [x,u,p,slack]
            self.optivar = [self.X, self.U, self.P, self.V]

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

            # Total Lagrange integrand
            lagrange = placeholders(self.lagrange)
            # Total Mayer term
            mayer = placeholders(self.mayer)

            ocp = self.ocp

            # At the moment, we don't detect any structure in cost...
            ocp.cost.cost_type = 'EXTERNAL'
            ocp.cost.cost_type_e = 'EXTERNAL'


            # .. except for slacks
            Qs, bs, lagrange = quadratic_coeff(lagrange, self.slack)
            Qs += DM.zeros(Sparsity.diag(self.slack.nnz()))
            assert Qs.sparsity().is_diag(), "Slacks cannot be mixed in objective"
            Qs_e, bs_e, mayer = quadratic_coeff(mayer, self.slack)
            assert Qs_e.nnz()==0, "Slacks in Mayer term not supported yet"
            Qs_e += DM.zeros(Sparsity.diag(self.slack.nnz()))
            assert Qs_e.sparsity().is_diag(), "Slacks cannot be mixed in objective"

            assert not depends_on(veccat(Qs, bs, Qs_e, bs_e), vertcat(self.ocp.model.x, self.ocp.model.u, self.slack)), \
              "Slack part of objective must be quadratic in slacks and depend only on parameters"

            ocp.model.cost_expr_ext_cost = lagrange
            ocp.model.cost_expr_ext_cost_e = mayer

            # For reference: https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf

            Jbx = []; lbx = []; ubx = []; Jsbx = []
            Jbu = []; lbu = []; ubu = []; Jsbu = []
            C = []; D = []; lg = []; ug = []; Jsg = []
            h = []; lbh = []; uh = []; Jsh = []

            h_e = []; lh_e = []; uh_e = []
            Jbx_e = []; lbx_e = []; ubx_e = []
            C_e = []; lg_e = []; ug_e = []

            x0_J = []
            x0_c = []


            def linear_coeffs(expr, *args):
                """ Multi-argument extesion to CasADi linear_coeff"""
                J,c = linear_coeff(canon, vcat(args))
                cs = np.cumsum([0]+[e.numel() for e in args])
                return tuple([J[:,cs[i]:cs[i+1]] for i in range(len(args))])+(c,)

            def legit_J(J):
                """
                Checks if J, a pre-multiplier for states and control, is of legitimate structure
                J must a slice of a permuted unit matrix.
                """
                try:
                    J = evalf(J)
                except:
                    return False
                if not np.all(np.array(J.nonzeros())==1):
                    return False
                # Each row must contain exactly 1 nonzero
                if not np.all(np.array(sum2(J))==1):
                    return False
                # Each column must contain at most 1 nonzero
                if not np.all(np.array(sum1(J))<=1):
                    return False
                return True

            def legit_Js(J):
                """
                Checks if J, a pre-multiplier for slacks, is of legitimate structure
                Empty rows are allowed
                """
                try:
                    check_Js(J)
                except:
                    return False
                return True

            def check_Js(J):
                """
                Checks if J, a pre-multiplier for slacks, is of legitimate structure
                Empty rows are allowed
                """
                try:
                    J = evalf(J)
                except:
                    raise Exception("Slack error")
                assert np.all(np.array(J.nonzeros())==1), "All nonzeros must be 1"
                # Check if slice of permutation of unit matrix
                assert np.all(np.array(sum2(J))<=1), "Each constraint can only depend on one slack at most"
                assert np.all(np.array(sum1(J))<=1), "Each constraint must depend on a unique slack, if any"

            def mark(slack, Js):
                assert np.all(np.array(slack[Js.sparsity().get_col()])==0)
                slack[Js.sparsity().get_col()] = 1
        
            def export_expr(m):
                if isinstance(m,list):
                    if len(m)==0:
                        return MX(0, 1)
                    else:
                        return vcat(m)
                return m
            
            def export_num(m):
                res=np.array(evalf(export_expr(m)))
                res[res==-inf] = -INF
                res[res==inf] = INF
                return res

            def export_num_vec(m):
                return np.array(evalf(export_expr(m))).reshape(-1)

            def export(m):
                return (export_expr(m),False)

            def export_vec(m):
                return (export_expr(m),True)


            # Flags to verify that each slack has a >=0 constraint
            slack_has_pos_const = DM.zeros(*self.slack.shape)

            # Flags to check sign
            slack_lower = DM.zeros(*self.slack.shape)
            slack_upper = DM.zeros(*self.slack.shape)

            # Process path constraints
            for c, meta, args in stage._constraints["control"]+stage._constraints["integrator"]:
                try:
                    c = substitute([placeholders(c)],self.raw,self.optivar)[0]
                    mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
                    lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)

                    # Check for infinities
                    try:
                        lb_inf = np.all(np.array(evalf(lb)==-inf))
                    except:
                        lb_inf = False
                    try:
                        ub_inf = np.all(np.array(evalf(ub)==inf))
                    except:
                        ub_inf = False
                    
                    # Slack positivity constraint
                    if depends_on(canon, self.slack) and not depends_on(canon, vertcat(self.ocp.model.x, self.ocp.model.u, self.ocp.model.p)):
                        J,c = linear_coeff(canon, self.slack)
                        is_perm = legit_Js(J)
                        lb_zero = np.all(np.array(evalf(lb-c)==0))
                        print(J,c,is_perm,lb_zero,ub_inf,lb,ub)
                        assert is_perm and lb_zero and ub_inf, "Only constraints allowed on slacks are '>=0'"
                        slack_has_pos_const[J.sparsity().get_col()] = 1
                        continue
                    
                    assert is_linear(canon, self.slack), "slacks can only enter linearly in constraints"

                    if is_linear(canon, vertcat(self.ocp.model.x,self.ocp.model.u)):
                        # Linear is states and controls
                        if not depends_on(canon, self.ocp.model.x):
                            # lbu <= Jbu u <= ubu
                            J, Js, c = linear_coeffs(canon, self.ocp.model.u, self.slack)
                            if legit_J(J):
                                Jbu.append(J)
                                lbu.append(lb-c)
                                ubu.append(ub-c)
                                if Js.nnz()>0:
                                    assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY]
                                
                                if not ub_inf: Js *= -1
                                check_Js(Js)
                                Jsbu.append(Js)
                                mark(slack_lower if ub_inf else slack_upper, Js)
                                continue
                            # Fallthrough
                        if not depends_on(canon, self.ocp.model.u):
                            # lbx <= Jbx x <= ubx
                            J, Js, c = linear_coeffs(canon, self.ocp.model.x, self.slack)
                            if legit_J(J):
                                Jbx.append(J)
                                lbx.append(lb-c)
                                ubx.append(ub-c)
                                if Js.nnz()>0:
                                    assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY]      
                                if not ub_inf: Js *= -1
                                check_Js(Js)
                                Jsbx.append(Js)
                                mark(slack_lower if ub_inf else slack_upper, Js)
                                continue
                            # Fallthrough
                        # lg <= Cx + Du <= ug
                        J1, J2, Js, c = linear_coeffs(canon, self.ocp.model.x,self.ocp.model.u, self.slack)
                        C.append(J1)
                        D.append(J2)
                        lg.append(lb-c)
                        ug.append(ub-c)
                        if Js.nnz()>0:
                            assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY]
                        if not ub_inf: Js *= -1
                        check_Js(Js)
                        Jsg.append(Js)
                        mark(slack_lower if ub_inf else slack_upper, Js)

                    else:
                        # Nonlinear constraints
                        Js, c = linear_coeffs(canon, self.slack)
                        # lbh <= h(x,u) <= uh
                        h.append(c)
                        lbh.append(lb)
                        uh.append(ub)
                        if Js.nnz()>0:
                            assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY]
                        if not ub_inf: Js *= -1
                        check_Js(Js)
                        Jsh.append(Js)
                        mark(slack_lower if ub_inf else slack_upper, Js)
                except Exception as e:
                    print(meta)
                    raise e
            
            # Lump together Js* across individual path constraint categories
            Jsbu = MX(0, 1) if len(Jsbu)==0 else vcat(Jsbu)
            Jsbx = MX(0, 1) if len(Jsbx)==0 else vcat(Jsbx)
            Jsg = MX(0, 1) if len(Jsg)==0 else vcat(Jsg)
            Jsh = MX(0, 1) if len(Jsh)==0 else vcat(Jsh)

            print("Jsbu", evalf(Jsbu), Jsbu.sparsity().get_col(), Jsbu.sparsity().row())
            print("Jsbx", evalf(Jsbx), Jsbx.sparsity().get_col(), Jsbx.sparsity().row())
            print("Jsg", evalf(Jsg), Jsg.sparsity().get_col(), Jsg.sparsity().row())
            print("Jsh", evalf(Jsh), Jsh.sparsity().get_col(), Jsh.sparsity().row())
            
            print("Qs", evalf(Qs))
            print("bs", evalf(bs), bs.shape)

            print(slack_lower,slack_upper)
            # Indices needed to pull lower and upper parts of Qs, bs apart
            li = sparsify(slack_lower).sparsity().row()
            ui = sparsify(slack_upper).sparsity().row()

            print(li, ui)

            ns = self.slack.nnz()
            print("ns", ns)

            Zl = MX(ns, ns)
            Zu = MX(ns, ns)
            zl = MX(ns, 1)
            zu = MX(ns, 1)
            Zl[li,li] = Qs[li,li]
            Zu[ui,ui] = Qs[ui,ui]
            zl[li,0] = bs[li,0]
            zu[ui,0] = bs[ui,0]

            print("Zl", evalf(Zl))
            print("Zu", evalf(Zu))
            print("zl", evalf(zl))
            print("zu", evalf(zu))

            # Columns give you the indices of used slacks
            bui = Jsbu.sparsity().get_col()
            bxi = Jsbx.sparsity().get_col()
            gi = Jsg.sparsity().get_col()
            hi = Jsh.sparsity().get_col()
            ni = bui+bxi+gi+hi

            # Re-order slacks according to (bu,bx,g,h)
            Zl = Zl[ni,ni]
            Zu = Zu[ni,ni]
            zl = zl[ni]
            zu = zu[ni]

            print("Zl", evalf(Zl))
            print("Zu", evalf(Zu))
            print("zl", evalf(zl))
            print("zu", evalf(zu))

            assert np.all(np.array(slack_has_pos_const)), "Only variables allowed are slacks (and they need '>=0' constraints)"

            # After re-ordering slacks, Js* become unit matrices interwoven with zero rows
            # But let's just work with idxs* directly
            self.ocp.constraints.idxsbu = np.array(Jsbu.sparsity().row())
            self.ocp.constraints.idxsbx = np.array(Jsbx.sparsity().row())
            self.ocp.constraints.idxsg = np.array(Jsg.sparsity().row())
            self.ocp.constraints.idxsh = np.array(Jsh.sparsity().row())

            # These should become parametric
            self.ocp.cost.Zl = export_num_vec(diag(Zl))
            self.ocp.cost.Zu = export_num_vec(diag(Zu))
            self.ocp.cost.zu = export_num_vec(zu)
            self.ocp.cost.zl = export_num_vec(zl)

            # Process point constraints
            # Probably should de-duplicate stuff wrrt path constraints code
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
                        lh_e.append(lb)
                        uh_e.append(ub)

            # Get x0 from constraint set
            x0 = mtimes(pinv(vcat(x0_J)), vcat(x0_c))

            ocp.constraints.constr_type = 'BGH'

            # No export_num here, let's do things parametrically
            self.m = m = OrderedDict()

            self.ocp.constraints.Jbx = export_num(Jbx)
            m["lbx"] = export_vec(lbx)
            m["ubx"] = export_vec(ubx)

            self.ocp.constraints.Jbu = export_num(Jbu)
            m["lbu"] = export_vec(lbu)
            m["ubu"] = export_vec(ubu)

            m["C"] = export(C)
            m["D"] = export(D)
            m["lg"] = export_vec(lg)
            m["ug"] = export_vec(ug)

            ocp.model.con_h_expr = export_expr(h)
            m["lh"] = export_vec(lbh)
            m["uh"] = export_vec(uh)

            if h_e:
                ocp.model.con_h_e_expr = export_expr(h_e)
                m["lh"] = export_vec(lh_e)
                m["uh"] = export_vec(uh_e)

            self.ocp.constraints.Jbx_e = export_num(Jbx_e)
            m["lbx_e"] = export_vec(lbx_e)
            m["ubx_e"] = export_vec(ubx_e)

            m["C_e"] = export(C_e)
            m["lg_e"] = export_vec(lg_e)
            m["ug_e"] = export_vec(ug_e)

            m["x0"] = export_vec(x0)

            args = [v[0] for v in m.values()]
            print([self.ocp.model.p],args,['p'],list(m.keys()))
            self.mmap = Function('mmap',[self.ocp.model.p],args,['p'],list(m.keys()))

            for k,v in stage._param_vals.items():
                self.set_value(stage, self, k, v)

            print("parameter array:", self.P0)
            self.ocp.parameter_values = np.array(self.P0).reshape((-1))

            res = self.mmap(p=self.P0)
            # Set matrices
            for k, (_,is_vec) in self.m.items():
                v = np.array(res[k])
                if is_vec:
                    v = v.reshape((-1))
                v[v==-inf] = -INF
                v[v==inf] = INF
                print(k,v)
                setattr(self.ocp.constraints, k, v)

            for k, v in self.args.items():
                setattr(ocp.solver_options, k, v)

            ocp.solver_options.tf = stage._T
            self.control_grid = casadi.linspace(MX(stage._t0), stage._t0+stage._T, self.N+1)

            self.ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + ocp.model.name + '.json')

    def set_value(self, stage, master, parameter, value):
            J = jacobian(parameter, self.ocp.model.p)
            self.P0[J.row()] = value
    
    def set_matrices(self):
        print("parameter array:", self.P0)
        self.ocp.parameter_values = np.array(self.P0).reshape((-1))

        res = self.mmap(p=self.P0)
        # Set matrices
        for k, (_,is_vec) in self.m.items():
            v = np.array(res[k])
            if is_vec:
                v = v.reshape((-1))
            v[v==-inf] = -INF
            v[v==inf] = INF
            for i in range(self.N):
                self.ocp_solver.constraints_set(i, k, v)

    def solve(self, stage):
        #self.set_matrices() # doesn't work yet
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