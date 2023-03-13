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

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from acados_template.utils import J_to_idx

from ...freetime import FreeTime
from ..method import ExternalMethod, legit_J, check_Js
from ...solution import OcpSolution

import numpy as np
from casadi import SX, Sparsity, MX, vcat, veccat, symvar, substitute, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag
import casadi
from ...casadi_helpers import DM2numpy, reshape_number
from collections import OrderedDict


INF = 1e5

def linear_coeffs(expr, *args):
    """ Multi-argument extesion to CasADi linear_coeff"""
    J,c = linear_coeff(expr, vcat(args))
    cs = np.cumsum([0]+[e.numel() for e in args])
    return tuple([J[:,cs[i]:cs[i+1]] for i in range(len(args))])+(c,)


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


class AcadosMethod(ExternalMethod):
    def __init__(self,**kwargs):
        ExternalMethod.__init__(self, **kwargs)

    def fill_placeholders_integral(self, phase, stage, expr, *args):
        if phase==1:
            I = stage.state()
            stage.set_der(I, expr)
            stage.subject_to(stage.at_t0(I)==0)
            return stage.at_tf(I)

    def transcribe_phase1(self, stage, **kwargs):
        self.stage = stage
        self.ocp = ocp = AcadosOcp()
        self.opti = Opti()



        self.X_gist = [MX.sym("X", stage.nx) for k in range(self.N+1)]
        self.U_gist = [MX.sym("U", stage.nu) for k in range(self.N)]

        import gc
        gc.collect()

        model = AcadosModel()

        f = stage._ode()

        if self.expand:
            model_x = SX.sym("x", stage.x.sparsity())
            model_u = SX.sym("u", stage.u.sparsity())
            model_p = SX.sym("p", stage.p.sparsity())
            model_xdot = SX.sym("xdot", stage.x.sparsity())

            tT = Function('temp',[stage.x,stage.u,stage.p],[self.t, self.T])
            model_t, model_T = tT(model_x, model_u, model_p)
        else:
            model_x = stage.x
            model_u = stage.u
            model_p = stage.p
            model_xdot = MX.sym("xdot", stage.nx)
            model_t = self.t
            model_T = self.T

        res = f(x=model_x, u=model_u, p=vertcat(model_p, DM.nan(stage.v.shape)), t=model_t)
        if isinstance(stage._T,FreeTime):
            f_expl = model_T*res["ode"]
        else:
            f_expl = res["ode"]

        model.f_impl_expr = model_xdot-f_expl
        model.f_expl_expr = f_expl
        model.x = model_x
        model.xdot = model_xdot
        model.u = model_u
        model.p = model_p
        self.P0 = DM.zeros(stage.np)

        slack = MX(0, 1) if len(stage.variables['control'])==0 else vvcat(stage.variables['control'])
        slack_e = MX(0, 1) if len(stage.variables[''])==0 else vvcat(stage.variables[''])

        self.slack = slack
        self.slack_e = slack_e
        self.slacks = vertcat(slack, slack_e)

        self.X = self.opti.variable(*stage.x.shape)
        self.U = self.opti.variable(*stage.u.shape)
        self.P = self.opti.parameter(*stage.p.shape)
        self.S = self.opti.variable(*slack.shape)
        self.Se = self.opti.variable(*slack_e.shape)
        self.t = self.opti.parameter()

        self.raw = [stage.x,stage.u,stage.p,slack,slack_e,stage.t]
        self.optivar = [self.X, self.U, self.P, self.S, self.Se, self.t]

        #self.time_grid = self.grid(stage._t0, stage._T, self.N)
        self.normalized_time_grid = self.grid(0.0, 1.0, self.N)
        self.time_grid = self.normalized_time_grid
        if not isinstance(stage._T, FreeTime): self.time_grid*= stage._T
        if not isinstance(stage._t0, FreeTime): self.time_grid+= stage._t0
        self.control_grid = MX(stage.t0 + self.normalized_time_grid*stage.T).T

        inits = []
        inits.append((stage.T, stage._T.T_init if isinstance(stage._T, FreeTime) else stage._T))
        inits.append((stage.t0, stage._t0.T_init if isinstance(stage._t0, FreeTime) else stage._t0))

        self.control_grid_init = evalf(substitute([self.control_grid], [a for a,b in inits],[b for a,b in inits])[0])

        #self.control_grid = self.normalized_time_grid

        model.name = "rockit_model"

        ocp.model = model

        # set dimensions
        ocp.dims.N = self.N

        var_lagrange = []
        var_mayer = []
        obj = MX(stage._objective)
        for e in symvar(obj):
            if "sum_control" in e.name():
                var_lagrange.append(e)
            else:#elif "at_tf" in e.name():
                var_mayer.append(e)
            #else:
            #    raise Exception("Unknown element in objective: %s" % str(e))

        self.lagrange = substitute([obj], var_mayer, [DM.zeros(e.shape) for e in var_mayer])[0]
        self.mayer = substitute([obj], var_lagrange, [DM.zeros(e.shape) for e in var_lagrange])[0]


        self.initial_conditions = []
        self.final_conditions = []

    def transcribe_phase2(self, stage, **kwargs):
        opti_advanced = self.opti.advanced
        placeholders = kwargs["placeholders"]

        # Total Lagrange integrand
        lagrange = placeholders(self.lagrange,preference=['expose'])*self.N
        # Total Mayer term
        mayer = placeholders(self.mayer,preference=['expose'])

        assert not depends_on(mayer, stage.u), "Mayer term of objective may not depend on controls"

        ocp = self.ocp

        # At the moment, we don't detect any structure in cost...
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # .. except for slacks
        assert not depends_on(lagrange, self.slack_e), "Lagrange term may not depend on a non-signal slack"
        assert not depends_on(mayer, self.slack), "Mayer term may not depend on a signal slack"
        Qs, bs, lagrange = quadratic_coeff(lagrange, self.slack)
        Qs += DM.zeros(Sparsity.diag(Qs.shape[0]))
        assert Qs.sparsity().is_diag(), "Slacks cannot be mixed in Lagrange objective"
        Qs_e, bs_e, mayer = quadratic_coeff(mayer, vertcat(self.slack,self.slack_e))
        Qs_e += DM.zeros(Sparsity.diag(Qs_e.shape[0]))
        assert Qs_e.sparsity().is_diag(), "Slacks cannot be mixed in Mayer objective"

        assert not depends_on(veccat(Qs, bs, Qs_e, bs_e), vertcat(stage.x, stage.u, self.slack, self.slack_e)), \
            "Slack part of objective must be quadratic in slacks and depend only on parameters"

        ocp.model.cost_expr_ext_cost = lagrange
        ocp.model.cost_expr_ext_cost_e = mayer

        #import ipdb
        #ipdb.set_trace()
        #raise Exception("")

        # For reference: https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf

        Jbx = []; lbx = []; ubx = []; Jsbx = []
        Jbu = []; lbu = []; ubu = []; Jsbu = []
        C = []; D = []; lg = []; ug = []; Jsg = []
        h = []; lh = []; uh = []; Jsh = []

        h_e = []; lh_e = []; uh_e = []; Jsh_e = []
        Jbx_e = []; lbx_e = []; ubx_e = []; Jsbx_e = []
        C_e = []; lg_e = []; ug_e = []; Jsg_e = []

        lbx_0 = [];ubx_0 = []
        Jbx_0 = [];Jbxe_0 = []



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
            if np.any(res==-inf) or np.any(res==inf):
                print("WARNING: Double-sided constraints are much preferred. Replaced inf with %f." % INF)
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
                c = substitute([placeholders(c,preference=['expose'])],self.raw,self.optivar)[0]
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

                assert not depends_on(canon, self.slack_e), "Path constraints may only have signal slacks"
                
                # Slack positivity constraint
                if depends_on(canon, self.slack) and not depends_on(canon, vertcat(stage.x, stage.u, stage.p)):
                    J,c = linear_coeff(canon, self.slack)
                    is_perm = legit_Js(J)
                    lb_zero = np.all(np.array(evalf(lb-c)==0))
                    assert is_perm and lb_zero and ub_inf, "Only constraints allowed on slacks are '>=0'"
                    slack_has_pos_const[J.sparsity().get_col()] = 1
                    continue
                
                assert is_linear(canon, self.slack), "slacks can only enter linearly in constraints"

                if is_linear(canon, vertcat(stage.x,stage.u)):
                    # Linear is states and controls
                    if not depends_on(canon, stage.x):
                        # lbu <= Jbu u <= ubu
                        J, Js, c = linear_coeffs(canon, stage.u, self.slack)
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
                    if not depends_on(canon, stage.u):
                        # lbx <= Jbx x <= ubx
                        J, Js, c = linear_coeffs(canon, stage.x, self.slack)
                        if legit_J(J):
                            Jbx.append(J)
                            lbx.append(lb-c)
                            ubx.append(ub-c)
                            if args["include_last"]:
                                assert not depends_on(canon, self.slack), " lbx <= Jbx x <= ubx does not support slacks for qualifier include_last=True."
                                Jbx_e.append(J)
                                lbx_e.append(lb-c)
                                ubx_e.append(ub-c)
                            if args["include_first"]:
                                assert not depends_on(canon, self.slack), "lbx <= Jbx x <= ubx does not support slacks for qualifier include_first=True."
                                Jbx_0.append(J)
                                lbx_0.append(lb-c)
                                ubx_0.append(ub-c)
                                Jbxe_0.append( (mc.type == casadi.OPTI_EQUALITY) * J)
                            if Js.nnz()>0:
                                assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY]      
                            if not ub_inf: Js *= -1
                            check_Js(Js)
                            Jsbx.append(Js)
                            mark(slack_lower if ub_inf else slack_upper, Js)
                            continue
                        # Fallthrough
                    # lg <= Cx + Du <= ug
                    J1, J2, Js, c = linear_coeffs(canon, stage.x,stage.u, self.slack)
                    C.append(J1)
                    D.append(J2)
                    lg.append(lb-c)
                    ug.append(ub-c)
                    if args["include_last"]:
                        assert not depends_on(canon, self.slack), "lg <= Cx + Du <= ug does not support slacks for qualifier include_last=True.\n" + str(meta)
                        if J2.nnz()>0:
                            raise Exception("lg <= Cx + Du <= ug only supported for qualifier include_last=False.\n"  + str(meta))
                        else:
                            C_e.append(J1)
                            lg_e.append(lb-c)
                            ug_e.append(ub-c)
                    if not args["include_first"]:
                        raise Exception("lg <= Cx + Du <= ug only supported for qualifier include_first=True.\n" + str(meta))
                    if Js.nnz()>0:
                        assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY]
                    if not ub_inf: Js *= -1
                    check_Js(Js)
                    Jsg.append(Js)
                    mark(slack_lower if ub_inf else slack_upper, Js)

                else:
                    # Nonlinear constraints
                    Js, c = linear_coeffs(canon, self.slack)
                    # lh <= h(x,u) <= uh
                    h.append(c)
                    lh.append(lb)
                    uh.append(ub)
                    if args["include_last"]:
                        if depends_on(canon, stage.u):
                            if args["include_last"]!="auto":
                                raise Exception("lh <= h(x,u) <= uh only supported for qualifier include_last=False.\n" + str(meta))
                        else:
                            assert not depends_on(canon, self.slack), "lh <= h(x,u) <= uh does not support slacks for qualifier include_last=True."
                            h_e.append(c)
                            lh_e.append(lb)
                            uh_e.append(ub)
                    if not args["include_first"]:
                        raise Exception("lh <= h(x,u) <= uh only supported for qualifier include_first=True.\n" + str(meta))                            
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

        # Indices needed to pull lower and upper parts of Qs, bs apart
        li = sparsify(slack_lower).sparsity().row()
        ui = sparsify(slack_upper).sparsity().row()

        ns = self.slack.nnz()

        Zl = MX(ns, ns)
        Zu = MX(ns, ns)
        zl = MX(ns, 1)
        zu = MX(ns, 1)
        Zl[li,li] = Qs[li,li]
        Zu[ui,ui] = Qs[ui,ui]
        zl[li,0] = bs[li,0]
        zu[ui,0] = bs[ui,0]

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

        # Flags to verify that each slack has a >=0 constraint
        slack_e_has_pos_const = DM.zeros(*self.slack_e.shape)

        # Flags to check sign
        slack_e_lower = DM.zeros(*self.slack_e.shape)
        slack_e_upper = DM.zeros(*self.slack_e.shape)

        # Process point constraints
        # Probably should de-duplicate stuff wrt path constraints code
        for c, meta, _ in stage._constraints["point"]:
            try:
                # Make sure you resolve u to r_at_t0/r_at_tf
                c = placeholders(c,max_phase=1)
                has_t0 = 'r_at_t0' in [a.name() for a in symvar(c)]
                has_tf = 'r_at_tf' in [a.name() for a in symvar(c)]

                cb = c
                c = substitute([placeholders(c,preference='expose')],self.raw,self.optivar)[0]
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

                if has_t0 and has_tf:
                    raise Exception("Constraints mixing value at t0 and tf not allowed.")
                
                assert not depends_on(canon, self.slack), "Path constraints may only have non-signal slacks"

                if has_t0:
                    assert not depends_on(canon, self.slack_e), "Initial constraints may not have slacks."
                
                # Slack positivity constraint
                if depends_on(canon, self.slack_e) and not depends_on(canon, vertcat(stage.x, stage.u, stage.p)):
                    J,c = linear_coeff(canon, self.slack_e)
                    is_perm = legit_Js(J)
                    lb_zero = np.all(np.array(evalf(lb-c)==0))
                    assert is_perm and lb_zero and ub_inf, "Only constraints allowed on slacks are '>=0'"
                    slack_e_has_pos_const[J.sparsity().get_col()] = 1
                    continue
                
                assert is_linear(canon, self.slack_e), "slacks can only enter linearly in constraints"

                if has_t0:
                    # t0
                    check = is_linear(canon, stage.x)
                    check = check and not depends_on(canon, stage.u)
                    assert check, "at t=t0, only equality constraints on x are allowed. Got '%s'" % str(c)

                    J,c = linear_coeff(canon, stage.x)
                    Jbx_0.append(J)
                    lbx_0.append(lb-c)
                    ubx_0.append(ub-c)

                    Jbxe_0.append( (mc.type == casadi.OPTI_EQUALITY) * J)
                else:
                    # tf
                    assert not depends_on(canon,stage.u), "Terminal constraints cannot depend on u"
                    if is_linear(canon, stage.x):
                        # lbx <= Jbx x <= ubx
                        J,Js,c = linear_coeffs(canon, stage.x, self.slack_e)
                        if legit_J(J):
                            Jbx_e.append(J)
                            lbx_e.append(lb-c)
                            ubx_e.append(ub-c)

                            if not ub_inf: Js *= -1
                            check_Js(Js)
                            Jsbx_e.append(Js)
                            mark(slack_e_lower if ub_inf else slack_e_upper, Js)

                            continue
                        # lg <= Cx <= ug
                        J, Js, c = linear_coeffs(canon, stage.x, self.slack_e)
                        C_e.append(J)
                        lg_e.append(lb-c)
                        ug_e.append(ub-c)
                        if not ub_inf: Js *= -1
                        check_Js(Js)
                        Jsg_e.append(Js)
                        mark(slack_e_lower if ub_inf else slack_e_upper, Js)
                    else:
                        Js, c = linear_coeff(canon, self.slack_e)
                        # lh <= h(x,u) <= uh
                        h_e.append(c)
                        lh_e.append(lb)
                        uh_e.append(ub)
                        if not ub_inf: Js *= -1
                        check_Js(Js)
                        Jsh_e.append(Js)
                        mark(slack_e_lower if ub_inf else slack_e_upper, Js)
            except Exception as e:
                print(meta)
                raise e

        # Lump together Js* across individual path constraint categories
        Jsbx_e = MX(0, 1) if len(Jsbx_e)==0 else vcat(Jsbx_e)
        Jsg_e = MX(0, 1) if len(Jsg_e)==0 else vcat(Jsg_e)
        Jsh_e = MX(0, 1) if len(Jsh_e)==0 else vcat(Jsh_e)

        # Indices needed to pull lower and upper parts of Qs, bs apart
        li = sparsify(slack_e_lower).sparsity().row()
        ui = sparsify(slack_e_upper).sparsity().row()

        ns = self.slack_e.nnz()

        Zl = MX(ns, ns)
        Zu = MX(ns, ns)
        zl = MX(ns, 1)
        zu = MX(ns, 1)
        Zl[li,li] = Qs_e[li,li]
        Zu[ui,ui] = Qs_e[ui,ui]
        zl[li,0] = bs_e[li,0]
        zu[ui,0] = bs_e[ui,0]

        bxi = Jsbx_e.sparsity().get_col()
        gi = Jsg_e.sparsity().get_col()
        hi = Jsh_e.sparsity().get_col()
        ni = bxi+gi+hi

        # Re-order slacks according to (bu,bx,g,h)
        Zl = Zl[ni,ni]
        Zu = Zu[ni,ni]
        zl = zl[ni]
        zu = zu[ni]

        assert np.all(np.array(slack_e_has_pos_const)), "Only variables allowed are slacks (and they need '>=0' constraints)"

        # After re-ordering slacks, Js* become unit matrices interwoven with zero rows
        # But let's just work with idxs* directly
        self.ocp.constraints.idxsbx_e = np.array(Jsbx_e.sparsity().row())
        self.ocp.constraints.idxsg_e = np.array(Jsg_e.sparsity().row())
        self.ocp.constraints.idxsh_e = np.array(Jsh_e.sparsity().row())

        # These should become parametric
        self.ocp.cost.Zl_e = export_num_vec(diag(Zl))
        self.ocp.cost.Zu_e = export_num_vec(diag(Zu))
        self.ocp.cost.zu_e = export_num_vec(zu)
        self.ocp.cost.zl_e = export_num_vec(zl)

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
        m["lh"] = export_vec(lh)
        m["uh"] = export_vec(uh)

        if h_e:
            ocp.model.con_h_expr_e = export_expr(h_e)
            m["lh_e"] = export_vec(lh_e)
            m["uh_e"] = export_vec(uh_e)

        self.ocp.constraints.Jbx_e = export_num(Jbx_e)
        m["lbx_e"] = export_vec(lbx_e)
        m["ubx_e"] = export_vec(ubx_e)

        m["C_e"] = export(C_e)
        m["lg_e"] = export_vec(lg_e)
        m["ug_e"] = export_vec(ug_e)

        self.ocp.constraints.Jbx_0 = export_num(Jbx_0)

        def None2Empty(a):
            if a is None:
                return MX(0, 1)
            else:
                return a

        if self.expand:
            temp = Function('temp', [stage.x, stage.u, stage.p], [None2Empty(ocp.model.cost_expr_ext_cost), None2Empty(ocp.model.cost_expr_ext_cost_e),  None2Empty(ocp.model.con_h_expr_e), None2Empty(ocp.model.con_h_expr)])
            [ocp.model.cost_expr_ext_cost, ocp.model.cost_expr_ext_cost_e, ocp.model.con_h_expr_e, ocp.model.con_h_expr] = temp(self.ocp.model.x, self.ocp.model.u, self.ocp.model.p)

        # Issue: if you indicate some states for equality, they are eliminated,
        # and the presence of other bounds on that state are problematic
        #
        # Filter idxbxe_0
        idxbxe_0 = np.sum(export_num(Jbxe_0),1).nonzero()[0]
        idx = list(J_to_idx(export_num(Jbx_0)))
        idxbxe_0_filtered = []
        for e in idxbxe_0:
            if idx.count(e)==1:
                idxbxe_0_filtered.append(e)
        # Still not safe
        idxbxe_0_filtered = []
        self.ocp.constraints.idxbxe_0 = np.array(idxbxe_0_filtered)
        m["lbx_0"] = export_vec(lbx_0)
        m["ubx_0"] = export_vec(ubx_0)

        args = [v[0] for v in m.values()]
        self.mmap = Function('mmap',[stage.p,stage.t],args,['p','t'],list(m.keys()))

        for k,v in stage._param_vals.items():
            self.set_value(stage, self, k, v)

        self.ocp.parameter_values = np.array(self.P0).reshape((-1))

        res = self.mmap(p=self.P0,t=0)
        # Set matrices
        for k, (_,is_vec) in self.m.items():
            v = np.array(res[k])
            if is_vec:
                v = v.reshape((-1))
            v[v==-inf] = -INF
            v[v==inf] = INF
            setattr(self.ocp.constraints, k, v)

        for k, v in self.args.items():
            setattr(ocp.solver_options, k, v)


        ocp.solver_options.tf = 1 if isinstance(stage._T, FreeTime) else stage._T
        ocp.solver_options.qp_solver_iter_max = 1000
        #ocp.solver_options.tol = 1e-8
        #ocp.solver_options.print_level = 15
        ocp.solver_options.shooting_nodes =np.ndarray.flatten(np.array(self.time_grid))
        # AcadosOcpOptions

        # By-pass acados's heuristic to check lbx==ubx numerically
        ocp.dims.nbxe_0 = self.ocp.constraints.idxbxe_0.shape[0]

        self.ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + ocp.model.name + '.json')

        #self.ocp_solver.options_set("step_length", 0.1)
        #self.ocp_solver.options_set("globalization", "fixed_step") # fixed_step, merit_backtracking
        if self.linesearch:
            self.ocp_solver.options_set("globalization", "merit_backtracking")

        X0 = DM.zeros(ocp.dims.nx, self.N+1)
        U0 = DM.zeros(ocp.dims.nu, self.N)


        for var, expr in stage._initial.items():
            if depends_on(expr, stage.t):
                expr = evalf(substitute(expr,stage.t, self.control_grid_init))

            var = substitute([var],self.raw,self.optivar)[0]
            Jx, Ju, r = linear_coeffs(var,self.X, self.U)
            Jx = evalf(Jx)
            Ju = evalf(Ju)
            r = evalf(r)
            assert r.is_zero()
            check_Js(Jx)
            check_Js(Ju)
            assert Jx.nnz()==0 or Ju.nnz()==0
            expr = reshape_number(var, expr)
            is_matrix = False
            if expr.shape[1]!= var.shape[1]:
                if expr.shape[1]==self.N*var.shape[1] or expr.shape[1]==(self.N+1)*var.shape[1]:
                    is_matrix = True
                else:
                    raise Exception("Initial guess of wrong shape")
            assert expr.shape[0]==var.shape[0]
            if Jx.sparsity().get_col():
                for k in range(self.N+1):
                    X0[Jx.sparsity().get_col(),k] = expr[Jx.row(),k if is_matrix else 0]
            if Ju.sparsity().get_col():
                for k in range(self.N):
                    U0[Ju.sparsity().get_col(),k] = expr[Ju.row(),k if is_matrix else 0]

        X0 = np.array(X0)
        U0 = np.array(U0)

        for k in range(self.N+1):
            self.ocp_solver.set(k, "x", X0[:,k])

        for k in range(self.N):
            self.ocp_solver.set(k, "u", U0[:,k])

    def set_matrices(self):
        self.ocp.parameter_values = np.array(self.P0).reshape((-1))

        res = self.mmap(p=self.P0,t=self.normalized_time_grid.T)
        # Set matrices
        for k, (_,is_vec) in self.m.items():
            v = np.array(res[k])
            v[v==-inf] = -INF
            v[v==inf] = INF
            if k=="lbx_e":
                self.ocp_solver.constraints_set(self.N, "lbx", v[:,-1])
            elif k=="ubx_e":
                self.ocp_solver.constraints_set(self.N, "ubx", v[:,-1])
            elif k=="C_e":
                if v.shape[0]*v.shape[1]>0:
                    self.ocp_solver.constraints_set(self.N, "C", v[:,:self.mmap.size2_out("C_e")])
            elif k=="lg_e":
                self.ocp_solver.constraints_set(self.N, "lg", v[:,-1])
            elif k=="ug_e":
                self.ocp_solver.constraints_set(self.N, "ug", v[:,-1])
            elif k=="lh_e":
                self.ocp_solver.constraints_set(self.N, "lh", v[:,-1])
            elif k=="uh_e":
                self.ocp_solver.constraints_set(self.N, "uh", v[:,-1])
            elif k=="lbx_0":
                self.ocp_solver.constraints_set(0, "lbx", v[:,0])
            elif k=="ubx_0":
                self.ocp_solver.constraints_set(0, "ubx", v[:,0])
            else:
                for i in range(self.N):
                    if i==0 and k in ["lbx","ubx"]: continue
                    stride = self.mmap.size2_out(k)
                    if v.shape[0]>0:
                        e = v[:,i*stride:(i+1)*stride]
                        if is_vec:
                            e = e.reshape((-1))
                        self.ocp_solver.constraints_set(i, k, e, api='new')

    def solve(self, stage):
        self.set_matrices()
        status = self.ocp_solver.solve()
        self.ocp_solver.print_statistics()
        x = [self.ocp_solver.get(i, "x") for i in range(self.N+1)]
        u = [self.ocp_solver.get(i, "u") for i in range(self.N)]
        return OcpSolution(SolWrapper(self, vcat(x), vcat(u)), stage)

    def eval(self, stage, expr):
        return expr
        
    def eval_at_control(self, stage, expr, k):
        placeholders = stage.placeholders_transcribed
        expr = placeholders(expr,max_phase=1)
        ks = [stage.x,stage.u]
        vs = [self.X_gist[k], self.U_gist[min(k, self.N-1)]]
        if not self.t_state:
            ks += [stage.t]
            vs += [self.control_grid[k]]
        return substitute([expr],ks,vs)[0]

class SolWrapper:
    def __init__(self, method, x, u):
        self.method = method
        self.x = x
        self.u = u

    def value(self, expr, *args,**kwargs):
        placeholders = self.method.stage.placeholders_transcribed
        ret = evalf(substitute([placeholders(expr)],[self.method.gist],[vertcat(self.x, self.u)])[0])
        return ret.toarray(simplify=True)