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
import os
import shutil
import subprocess
import casadi as ca
import tempfile

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


"""
    def __init__(self,**kwargs):
        ExternalMethod.__init__(self, **kwargs)
        self.build_dir = tempfile.TemporaryDirectory(prefix="acados_rockit_")
        self.build_dir_abs = self.build_dir.name
        
    def __del__(self):
        self.build_dir.cleanup()
"""

def recursive_overwrite(src, dest, ignore=None):
    if os.path.isdir(src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        if ignore is not None:
            ignored = ignore(src, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                recursive_overwrite(os.path.join(src, f), 
                                    os.path.join(dest, f), 
                                    ignore)
    else:
        shutil.copyfile(src, dest)

class AcadosMethod(ExternalMethod):
    def __init__(self,feasibility_problem=False,acados_options=None,model_name="rockit_model",**kwargs):
        ExternalMethod.__init__(self, **kwargs)
        #self.build_dir_abs = "./build_acados_rockit"
        self.build_dir_abs = "./foobar"

        self.model_name = model_name

        self.feasibility_problem = feasibility_problem
        self.acados_options = {} if acados_options is None else acados_options

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


        self.X_gist = [ca.vvcat([MX.sym("Xg", s.size1(), s.size2()) for s in stage.states]) for k in range(self.N+1)]
        self.U_gist = [ca.vvcat([MX.sym("Ug", s.size1(), s.size2()) for s in stage.controls]) for k in range(self.N)]
        self.P_stage_gist = [ca.vvcat([MX.sym("stage_param%d" % k, s.size1(), s.size2()) for s in  self.p_local]) for k in range(self.N+1)]
        self.P_global_gist = ca.vvcat([MX.sym("Vg", s.size1(), s.size2()) for s in stage.parameters['']])
        self.T_gist = MX.sym("Tg")


        self.gist_parts = []
        self.gist_parts.append((self.X_gist, stage.x, "local"))
        self.gist_parts.append((self.U_gist, stage.u, "local"))
        self.gist_parts.append((self.P_global_gist, self.p_global_cat, "global"))
        self.gist_parts.append((self.P_stage_gist, self.p_local_cat, "local"))
        self.gist_parts.append((self.T_gist, stage.T, "global"))

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

        slack_bounds = stage.variables['']
        slack_0_index = set()
        slack_e_index = set()

        for c, meta, _ in stage._constraints["point"]:
            sv = symvar(c)
            case_0 = np.any(["_t0" in e.name() for e in sv])
            case_e = np.any(["_tf" in e.name() for e in sv])
            assert not (case_0 and case_e), "Point constraints must be either at t0 or tf"

            for i,s_b in enumerate(slack_bounds):
                if np.any([ca.is_equal(s_b,e) for e in sv]):
                    if case_0:
                        assert i not in slack_0_index, "Slack can only be used once in a point constraint"
                        slack_0_index.add(i)
                    elif case_e:
                        assert i not in slack_e_index, "Slack can only be used once in a point constraint"
                        slack_e_index.add(i)
        slack_0 = [slack_bounds[i] for i in sorted(slack_0_index)]
        slack_e = [slack_bounds[i] for i in sorted(slack_e_index)]

        slack_0 = MX(0, 1) if len(slack_0)==0 else vvcat(slack_0)
        slack_e = MX(0, 1) if len(slack_e)==0 else vvcat(slack_e)

        self.slack = slack
        self.slack_0 = slack_0
        self.slack_e = slack_e

        self.X = self.opti.variable(*stage.x.shape)
        self.U = self.opti.variable(*stage.u.shape)
        self.P = self.opti.parameter(*stage.p.shape)
        self.S = self.opti.variable(*slack.shape)
        self.Se = self.opti.variable(*slack_e.shape)
        self.S0 = self.opti.variable(*slack_0.shape)
        self.t = self.opti.parameter()

        self.raw = [stage.x,stage.u,stage.p,slack,slack_0,slack_e,stage.t]
        self.optivar = [self.X, self.U, self.P, self.S, self.S0, self.Se, self.t]

        #self.time_grid = self.grid(stage._t0, stage._T, self.N)
        self.normalized_time_grid = self.grid(0.0, 1.0, self.N)
        self.time_grid = self.normalized_time_grid
        if not isinstance(stage._T, FreeTime): self.time_grid*= stage._T
        if not isinstance(stage._t0, FreeTime): self.time_grid+= stage._t0
        self.control_grid = MX(stage.t0 + self.normalized_time_grid*stage.T).T

        inits = []
        inits.append((stage.T, stage._T.T_init if isinstance(stage._T, FreeTime) else stage._T))
        inits.append((stage.t0, stage._t0.T_init if isinstance(stage._t0, FreeTime) else stage._t0))

        self.inits = inits

        self.control_grid_init = evalf(substitute([self.control_grid], [a for a,b in inits],[b for a,b in inits])[0])

        #self.control_grid = self.normalized_time_grid

        model.name = self.model_name

        ocp.model = model

        # set dimensions
        ocp.dims.N = self.N

        lagrange = MX(0)
        mayer = MX(0)
        mayer0 = MX(0)
        
        def split_sum(e):
            if e.op()==ca.OP_ADD:
                for p in split_sum(e.dep(0)):
                    yield p
                for p in split_sum(e.dep(1)):
                    yield p
            else:
                yield e

        for obj in split_sum(MX(stage._objective)):
            if "sum_control" in str(obj):
                lagrange += obj
            elif ca.depends_on(obj,slack_0) or "_at_t0" in str(obj):
                mayer0 += obj
            else:
                mayer += obj

        self.lagrange = lagrange
        self.mayer = mayer
        self.mayer0 = mayer0

        self.initial_conditions = []
        self.final_conditions = []

    def transcribe_phase2(self, stage, **kwargs):
        opti_advanced = self.opti.advanced
        placeholders = kwargs["placeholders"]

        # Total Lagrange integrand
        lagrange = placeholders(self.lagrange,preference=['expose'])*self.N
        # Total Mayer term
        mayer = placeholders(self.mayer,preference=['expose'])
        mayer0 = placeholders(self.mayer0,preference=['expose'])

        assert not depends_on(mayer, stage.u), "Mayer term of objective may not depend on controls"

        ocp = self.ocp

        # At the moment, we don't detect any structure in cost...
        #ocp.cost.cost_type_0 = 'EXTERNAL'
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # .. except for slacks
        assert not depends_on(lagrange, vertcat(self.slack_0,self.slack_e)), "Lagrange term may not depend on a non-signal slack"
        assert not depends_on(mayer, self.slack), "Mayer term may not depend on a signal slack"
        assert not depends_on(mayer0, self.slack), "Mayer0 term may not depend on a signal slack"
        assert not depends_on(mayer, self.slack_0), "Mayer term may not depend on a slack_0"
        assert not depends_on(mayer0, self.slack_e), "Mayer0 term may not depend on a slack_e"

        Qs, bs, lagrange = quadratic_coeff(lagrange, self.slack)
        Qs += DM.zeros(Sparsity.diag(Qs.shape[0]))
        assert Qs.sparsity().is_diag(), "Slacks cannot be mixed in Lagrange objective"
        Qs_e, bs_e, mayer = quadratic_coeff(mayer, vertcat(self.slack,self.slack_e))
        Qs_e += DM.zeros(Sparsity.diag(Qs_e.shape[0]))
        assert Qs_e.sparsity().is_diag(), "Slacks cannot be mixed in Mayer objective"

        Qs_0, bs_0, mayer0 = quadratic_coeff(mayer0, vertcat(self.slack,self.slack_0))
        Qs_0 += DM.zeros(Sparsity.diag(Qs_0.shape[0]))
        assert Qs_0.sparsity().is_diag(), "Slacks cannot be mixed in Mayer0 objective"

        assert not depends_on(veccat(Qs, bs, Qs_0, bs_0, Qs_e, bs_e), vertcat(stage.x, stage.u, self.slack, self.slack_0, self.slack_e)), \
            "Slack part of objective must be quadratic in slacks and depend only on parameters"

        #ocp.model.cost_expr_ext_cost_0 = mayer0
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
        h_0 = []; lh_0 = []; uh_0 = []; Jsh_0 = []
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
                    if args["include_first"]:
                        assert not depends_on(canon, self.slack), "lh <= h(x,u) <= uh does not support slacks for qualifier include_first=True."
                        h_0.append(c)
                        lh_0.append(lb)
                        uh_0.append(ub)
                    if args["include_first"]:
                        raise Exception("lh <= h(x,u) <= uh only supported for qualifier include_first=False.\n" + str(meta))                            
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

        ni0 = bui+gi
        Zl0 = Zl[ni0,ni0]
        Zu0 = Zu[ni0,ni0]
        zl0 = zl[ni0,0]
        zu0 = zu[ni0,0]

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
        slack_0_has_pos_const = DM.zeros(*self.slack_0.shape)

        # Flags to check sign
        slack_e_lower = DM.zeros(*self.slack_e.shape)
        slack_e_upper = DM.zeros(*self.slack_e.shape)
        slack_0_lower = DM.zeros(*self.slack_0.shape)
        slack_0_upper = DM.zeros(*self.slack_0.shape)

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
                    assert not depends_on(canon, self.slack_e), "Initial constraints may not have final slacks."
                if has_tf:
                    assert not depends_on(canon, self.slack_0), "Initial constraints may not have initial slacks."

                skip = False
                for slack, slack_has_pos_const in zip([self.slack_0,self.slack_e],[slack_0_has_pos_const,slack_e_has_pos_const]):
                    # Slack positivity constraint
                    if depends_on(canon, slack) and not depends_on(canon, vertcat(stage.x, stage.u, stage.p)):
                        J,c = linear_coeff(canon, slack)
                        is_perm = legit_Js(J)
                        lb_zero = np.all(np.array(evalf(lb-c)==0))
                        assert is_perm and lb_zero and ub_inf, "Only constraints allowed on slacks are '>=0'"
                        slack_has_pos_const[J.sparsity().get_col()] = 1
                        skip = True
                # Do not process these further
                if skip: continue

                if has_t0:
                    slack = self.slack_0
                if has_tf:
                    slack = self.slack_e

                
                assert is_linear(canon, slack), "slacks can only enter linearly in constraints"

                if has_t0:
                    if mc.type==casadi.OPTI_EQUALITY:
                        # t0
                        check = is_linear(canon, stage.x)
                        check = check and not depends_on(canon, stage.u)
                        assert check, "at t=t0, only equality constraints on x are allowed. Got '%s'" % str(c)

                        J,c = linear_coeff(canon, stage.x)
                        Jbx_0.append(J)
                        lbx_0.append(lb-c)
                        ubx_0.append(ub-c)

                        Jbxe_0.append( J)
                    else:
                        Js, c = linear_coeff(canon, self.slack_0)
                        # lh <= h(x,u) <= uh
                        h_0.append(c)
                        lh_0.append(lb)
                        uh_0.append(ub)
                        if not ub_inf: Js *= -1
                        check_Js(Js)
                        Jsh_e.append(Js)
                        mark(slack_0_lower if ub_inf else slack_0_upper, Js)
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


        # Lump together Js* across individual path constraint categories
        Jsh_0 = MX(0, 1) if len(Jsh_0)==0 else vcat(Jsh_0)

        # Indices needed to pull lower and upper parts of Qs, bs apart
        li = sparsify(slack_0_lower).sparsity().row()
        ui = sparsify(slack_0_upper).sparsity().row()

        ns = self.slack_0.nnz()

        Zl = MX(ns, ns)
        Zu = MX(ns, ns)
        zl = MX(ns, 1)
        zu = MX(ns, 1)
        Zl[li,li] = Qs_0[li,li]
        Zu[ui,ui] = Qs_0[ui,ui]
        zl[li,0] = bs_0[li,0]
        zu[ui,0] = bs_0[ui,0]

        hi = Jsh_0.sparsity().get_col()
        ni = hi

        # Re-order slacks according to (h)
        Zl = Zl[ni,ni]
        Zu = Zu[ni,ni]
        zl = zl[ni]
        zu = zu[ni]

        Zl = ca.diagcat(Zl0,Zl)
        Zu = ca.diagcat(Zu0,Zu)
        zl = ca.vertcat(zl0,zl)
        zu = ca.vertcat(zu0,zu)

        assert np.all(np.array(slack_0_has_pos_const)), "Only variables allowed are slacks (and they need '>=0' constraints)"

        # After re-ordering slacks, Js* become unit matrices interwoven with zero rows
        # But let's just work with idxs* directly

        self.ocp.constraints.idxsh_0 = np.array(Jsh_0.sparsity().row())

        # These should become parametric
        self.ocp.cost.Zl_0 = export_num_vec(diag(Zl))
        self.ocp.cost.Zu_0 = export_num_vec(diag(Zu))
        self.ocp.cost.zu_0 = export_num_vec(zu)
        self.ocp.cost.zl_0 = export_num_vec(zl)


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

        if h_0:
            ocp.model.con_h_expr_0 = export_expr(h_0)
            m["lh_0"] = export_vec(lh_0)
            m["uh_0"] = export_vec(uh_0)
        else:
            m["lh_0"] = export_vec(MX.zeros(0,1))
            m["uh_0"] =export_vec( MX.zeros(0,1))

        if h_e:
            ocp.model.con_h_expr_e = export_expr(h_e)
            m["lh_e"] = export_vec(lh_e)
            m["uh_e"] = export_vec(uh_e)
        else:
            m["lh_e"] = export_vec(MX.zeros(0,1))
            m["uh_e"] =export_vec( MX.zeros(0,1))


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
            temp = Function('temp', [stage.x, stage.u, stage.p], [None2Empty(ocp.model.cost_expr_ext_cost), None2Empty(ocp.model.cost_expr_ext_cost_e),  None2Empty(ocp.model.con_h_expr_0), None2Empty(ocp.model.con_h_expr_e), None2Empty(ocp.model.con_h_expr)])
            [ocp.model.cost_expr_ext_cost, ocp.model.cost_expr_ext_cost_e, ocp.model.con_h_expr_0, ocp.model.con_h_expr_e, ocp.model.con_h_expr] = temp(self.ocp.model.x, self.ocp.model.u, self.ocp.model.p)

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
        outputs = list(m.keys())
        self.mmap = Function('mmap',[self.p_global_cat,self.p_local_cat,stage.t],args,['p_global','p_local','t'],outputs)

        tgrid = ca.MX.sym("tgrid",1,self.N+1)
        P_local = [ca.MX.sym("P_local",self.p_local_cat.shape[0]) for i in range(self.N+1)]
        resv = [self.mmap(p_global=self.p_global_cat,p_local=p_local,t=t) for t,p_local in zip(ca.horzsplit(tgrid),P_local)]

        res = {}

        for k in m.keys():
            if k.endswith("_0"):
                res[k] = resv[0][k]
            elif k.endswith("_e"):
                res[k] = resv[-1][k]
            else:
                res[k] = ca.hcat([r[k] for r in resv])

        for k in m.keys():
            res[k] = ca.fmax(ca.fmin(res[k], INF),-INF)

        self.mmap_horizon = Function('mmap_horizon',[self.p_global_cat,ca.hcat(P_local),tgrid],[res[k] for k in outputs],['p_global','p_local','t'],outputs)

        for k,v in stage._param_vals.items():
            self.set_value(stage, self, k, v)

        self.ocp.parameter_values = np.zeros(stage.p.shape[0])

        res = self.mmap(p_global=self.p_global_value,p_local=self.p_local_value[:,0],t=0)
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

        for k,v in self.acados_options.items():
            setattr(ocp.solver_options, k, v)

        if self.feasibility_problem:
            ocp.translate_to_feasibility_problem()

        # By-pass acados's heuristic to check lbx==ubx numerically
        ocp.dims.nbxe_0 = self.ocp.constraints.idxbxe_0.shape[0]

        #self.ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + ocp.model.name + '.json')

        AcadosOcpSolver.generate(ocp, json_file = 'acados_ocp_' + ocp.model.name + '.json')

        c_generated_code = os.path.join(os.getcwd(), "c_generated_code")

        if not os.path.exists(os.path.join(self.build_dir_abs,"build")):
            try:
                # copy acados
                ACADOS_SOURCE_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep + "external"
                if 'ACADOS_SOURCE_DIR' in os.environ:
                    ACADOS_SOURCE_DIR = os.environ['ACADOS_SOURCE_DIR']
                recursive_overwrite(ACADOS_SOURCE_DIR, os.path.join(self.build_dir_abs,"acados"))
            except:
                pass
        recursive_overwrite(os.path.dirname(os.path.realpath(__file__)) + os.sep + "interface_generation",self.build_dir_abs)
        recursive_overwrite(c_generated_code,self.build_dir_abs)

        with open(os.path.join(self.build_dir_abs,"rockit_config.h"),"w") as out:

            for i in range(self.mmap_horizon.n_out()):
                k = self.mmap_horizon.name_out(i)
                if (k.endswith("_0") or k.endswith("_e")) and self.m[k][1]:
                    out.write(f"#define MMAP_{k.upper()}_SIZE {self.mmap_horizon.numel_out(i)}\n")
                else:
                    out.write(f"#define MMAP_{k.upper()}_SIZE1 {self.mmap_horizon.size1_out(i)}\n")
                    out.write(f"#define MMAP_{k.upper()}_SIZE2 {self.mmap_horizon.size2_out(i)}\n")
                    # Same but uppercase
            out.write(f"#define ROCKIT_N {self.N}\n")
            out.write(f"#define ROCKIT_X_SIZE1 {stage.nx}\n")
            out.write(f"#define ROCKIT_X_SIZE2 {self.N+1}\n")
            out.write(f"#define ROCKIT_U_SIZE1 {stage.nu}\n")
            out.write(f"#define ROCKIT_U_SIZE2 {self.N}\n")
            out.write(f"#define ROCKIT_P_LOCAL_SIZE1 {self.p_local_cat.shape[0]}\n")
            out.write(f"#define ROCKIT_P_LOCAL_SIZE2 {self.N+1}\n")
            out.write(f"#define ROCKIT_P_GLOBAL_SIZE1 {self.p_global_cat.shape[0]}\n")
            out.write(f"#define ROCKIT_P_GLOBAL_SIZE2 {self.p_global_cat.shape[1]}\n")

        if 0:
            with open(os.path.join(self.build_dir_abs,"after_init.c.in"), "w") as after_init:
                if self.linesearch:
                    after_init.write(f"""ocp_nlp_solver_opts_set(m->nlp_config, m->nlp_opts, "globalization","merit_backtracking");\n""")

                for k,v in self.acados_options.items():
                    if isinstance(v, bool):
                        after_init.write(f"""bool {k}={int(v)};ocp_nlp_solver_opts_set(m->nlp_config, m->nlp_opts, "{k}",&{k});\n""")
                    elif isinstance(v, str):
                        after_init.write(f"""ocp_nlp_solver_opts_set(m->nlp_config, m->nlp_opts, "{k}","{v}");\n""")
                    elif isinstance(v, bool):
                        after_init.write(f"""int {k}={v};ocp_nlp_solver_opts_set(m->nlp_config, m->nlp_opts, "{k}",&{k});\n""")
        assert subprocess.run(["cmake","-S", ".","-B", "build","-DCMAKE_BUILD_TYPE=Debug", "-DMODEL_NAME="+ self.model_name], cwd=self.build_dir_abs).returncode==0
        assert subprocess.run(["cmake","--build","build","--config","Debug"], cwd=self.build_dir_abs).returncode==0
        assert subprocess.run(["cmake","--install","build","--prefix","."], cwd=self.build_dir_abs).returncode==0
        
        self.acados_driver = ca.external("acados_driver", self.build_dir_abs + "/lib/libacados_driver.so")

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

        self.X0 = np.array(X0)
        self.U0 = np.array(U0)


    def to_function(self, stage, name, args, results, *margs):

        args = [stage.value(a) for a in args]

        controls = ca.hcat(self.U_gist)
        states = ca.hcat(self.X_gist)
        p_stage = ca.hcat(self.P_stage_gist)
        p_global = self.P_global_gist

        gist_list = self.U_gist+self.X_gist+self.P_stage_gist+[self.P_global_gist,self.T_gist]
        try:
            helper0 = Function("helper0", args, gist_list, {"allow_free":True})
        except:
            helper0 = Function("helper0", args, gist_list)
        if helper0.has_free():
            helper0_free = helper0.free_mx()
            [controls, states, p_stage, p_global, T_gist] = ca.substitute([controls, states, p_stage, p_global, self.T_gist], helper0_free, self.initial_value(stage, helper0_free))

        res = self.mmap_horizon(p_local=p_stage,p_global=p_global,t=self.normalized_time_grid.T)

        res["x0"] = states
        res["u0"] = controls

        res["p_local"] = p_stage
        res["p_global"] = p_global

        res = self.acados_driver(**res)

        arg_in = ca.substitute(results,gist_list, ca.horzsplit(res["u"])+ca.horzsplit(res["x"])+ca.horzsplit(p_stage)+[p_global,T_gist])

        ret = Function(name, args, arg_in, *margs)
        assert not ret.has_free()
        return ret
    
    def solve(self, stage):
        res = self.mmap_horizon(p_global=self.p_global_value,p_local=self.p_local_value,t=self.normalized_time_grid.T)

        res["p_local"] = self.p_local_value
        res["p_global"] = self.p_global_value
    
        res["x0"] = self.X0
        res["u0"] = self.U0
        ret = self.acados_driver(**res)

        return OcpSolution(SolWrapper(self, ca.vec(ret["x"]), ca.vec(ret["u"]), 0, self.p_global_value, self.p_local_value), stage)

    def initial_value(self, stage, expr):
        # check if expr is a list
        lst = isinstance(expr, list) 
        if not lst:
            expr = [expr]
        # expr = stage.value(expr)
        [_,states] = stage.sample(stage.x,grid='control')
        [_,controls] = stage.sample(stage.u,grid='control-')
        variables = stage.value(veccat(stage.variables['']))

        helper_in = [self.P_global_gist,ca.horzcat(*self.P_stage_gist),states,controls,variables,self.T_gist]
        helper = Function("helper", helper_in, expr)
        ret = [s.toarray(simplify=True) for s in helper.call([self.p_global_value, self.p_local_value, self.X0, self.U0, 0, self.inits[0][1]])]
        if not lst :
            return ret[0]
        return ret
class SolWrapper:
    def __init__(self, method, x, u, T, P_global, P_stage):
        self.method = method
        self.x = x
        self.u = u
        self.P_global = P_global
        self.P_stage = P_stage
        self.T = T

    def value(self, expr, *args,**kwargs):
        placeholders = self.method.stage.placeholders_transcribed
        ret = evalf(substitute([placeholders(expr)],[self.method.gist],[veccat(self.x, self.u, self.P_global, self.P_stage, self.T)])[0])
        return ret.toarray(simplify=True)