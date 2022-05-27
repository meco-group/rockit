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

from ...freetime import FreeTime
from ..method import ExternalMethod, linear_coeffs
from ...solution import OcpSolution

import numpy as np
from casadi import CodeGenerator, SX, Sparsity, MX, vcat, veccat, symvar, substitute, densify, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag
import casadi
from ...casadi_helpers import DM2numpy, reshape_number
from collections import OrderedDict

import subprocess
import os
from ctypes import *


"""
In general, the constraints should be formulated in such a way that there are no conflicts. However,
numerical difficulties can arise in some problems if constraints are formulated twice for the last point.
Therefore, GRAMPC does not evaluate the constraints g and h for the last trajectory point if terminal
constraints are defined, i.e. NgT + NhT > 0. In contrast, if no terminal constraints are defined, the
functions g and h are evaluated for all points. Note that the opposite behavior is easy to implement
by including g and h in the terminal constraints gT and hT .


"""


def format_float(e):
    return "%0.18f" % e

def strlist(a):
    elems = []
    for e in a:
        if isinstance(e,str):
            elems.append('"'+e+'"')
        elif isinstance(e,float):
            elems.append(format_float(e))
        else:
            elems.append(str(e))
    return ",".join(elems)
    
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



class GrampcMethod(ExternalMethod):
    def __init__(self,
    	Nhor = 20,           # Number of steps for the system integration
        MaxGradIter = 1000,   # Maximum number of gradient iterations
	    MaxMultIter = 1, # Maximum number of augmented Lagrangian iterations */
	    ShiftControl = "off", 
        LineSearchMax = 2.0,
	    LineSearchExpAutoFallback = "off",

        AugLagUpdateGradientRelTol = 1.0, 
	    ConstraintsAbsTol = 1e-4,

        PenaltyMax = 1e4,
	    PenaltyMin = 50.0,
	    PenaltyIncreaseFactor = 1.1,
	    PenaltyDecreaseFactor = 1.0,

        ConvergenceCheck = "on",
	    ConvergenceGradientRelTol = 1e-6,
        **kwargs):
        ExternalMethod.__init__(self, **kwargs)
        self.codegen_name = 'casadi_codegen'
        self.grampc_driver = 'grampc_driver'
        self.user = "((cs_struct*) userparam)"

        self.Nhor = Nhor
        self.MaxGradIter = MaxGradIter
        self.MaxMultIter = MaxMultIter
        self.ShiftControl = ShiftControl
        self.LineSearchMax = LineSearchMax
        self.LineSearchExpAutoFallback = LineSearchExpAutoFallback
        self.AugLagUpdateGradientRelTol = AugLagUpdateGradientRelTol
        self.ConstraintsAbsTol = ConstraintsAbsTol
        self.PenaltyMax = PenaltyMax
        self.PenaltyMin = PenaltyMin
        self.PenaltyIncreaseFactor = PenaltyIncreaseFactor
        self.PenaltyDecreaseFactor = PenaltyDecreaseFactor
        self.ConvergenceCheck = ConvergenceCheck
        self.ConvergenceGradientRelTol = ConvergenceGradientRelTol

    def fill_placeholders_integral(self, phase, stage, expr, *args):
        if phase==1:
            return expr

    def _register(self,fun_name,argtypes,restype):
        self.prefix=""
        fun = getattr(self.lib,self.prefix+fun_name)
        setattr(self,"_"+fun_name,fun)
        fun.argtypes = argtypes
        fun.restype = restype

    def gen_interface(self, f):
        f = f.expand()
        self.codegen.add(f)
        print(f)
        self.preamble.append(f"{f.name()}_incref();")
        self.preamble.append(f"{f.name()}_work(&sz_arg_local, &sz_res_local, &sz_iw_local, &sz_w_local);")
        self.preamble.append("if (sz_arg_local>sz_arg) sz_arg=sz_arg_local;")
        self.preamble.append("if (sz_res_local>sz_res) sz_res=sz_res_local;")
        self.preamble.append("if (sz_iw_local>sz_iw) sz_iw=sz_iw_local;")
        self.preamble.append("if (sz_w_local>sz_iw) sz_w=sz_w_local;")
        self.postamble.append(f"{f.name()}_decref();")

        scalar = lambda name: name in ["t","T"]

        args = [
                f"ctypeRNum {'' if scalar(f.name_in(i)) else '*'}{f.name_in(i)}"
                    for i in range(f.n_in())
                    if "p_fixed" not in f.name_in(i)]
        self.output_file.write(f"void {f.name()[3:]}(typeRNum *out, {', '.join(args)}, typeUSERPARAM *userparam) {{\n")
        self.output_file.write("  int mem;\n")
        for i in range(f.n_in()):
            e = f.name_in(i)
            if scalar(e):
                self.output_file.write(f"  {self.user}->arg[{i}] = &{e};\n")
            elif e=="p_fixed":
                self.output_file.write(f"  {self.user}->arg[{i}] = {self.user}->p;\n")
            else:
                self.output_file.write(f"  {self.user}->arg[{i}] = {e};\n")
        self.output_file.write(f"  {self.user}->res[0] = out;\n")
        self.output_file.write(f"  mem = {f.name()}_checkout();\n")
        self.output_file.write(f"  {f.name()}({self.user}->arg, {self.user}->res, {self.user}->iw, {self.user}->w, mem);\n")
        self.output_file.write(f"  {f.name()}_release(mem);\n")
        self.output_file.write(f"}}\n")

    def transcribe_phase1(self, stage, **kwargs):

        self.preamble = ["casadi_int sz_arg=0, sz_res=0, sz_iw=0, sz_w=0;",
                         "casadi_int sz_arg_local, sz_res_local, sz_iw_local, sz_w_local;",
                        ]
        self.postamble = []
        self.output_file = open(f"{self.grampc_driver}.c", "w")
        self.output_file.write(f"""
        #include "{self.codegen_name}.h"
        #include "grampc.h"

        typedef struct cs_struct_def {{
            const casadi_real** arg;
            casadi_real** res;
            casadi_int* iw;
            casadi_real* w;
            casadi_int stride;
            casadi_real* p;
            typeGRAMPC* grampc;
        }} cs_struct;
    """)
        self.stage = stage
        self.opti = Opti()

        self.X_gist = [MX.sym("X", stage.nx) for k in range(self.N+1)]
        self.U_gist = [MX.sym("U", stage.nu) for k in range(self.N)]

        f = stage._ode()
        options = {}
        options["with_header"] = True
        self.codegen = CodeGenerator(f"{self.codegen_name}.c", options)


        self.v = vvcat(stage.variables[''])


        assert f.numel_out("alg")==0
        assert f.numel_out("quad")==0
        ffct = Function("cs_ffct", [stage.t, stage.x, stage.u, self.v, stage.p], [ densify(f(x=stage.x, u=stage.u, p=stage.p, t=stage.t)["ode"])],['t','x','u','p','p_fixed'],['out'])
        self.gen_interface(ffct)
        self.gen_interface(ffct.factory("cs_dfdx_vec",["t","x","adj:out","u","p","p_fixed"],["densify:adj:x"]))
        self.gen_interface(ffct.factory("cs_dfdu_vec",["t","x","adj:out","u","p","p_fixed"],["densify:adj:u"]))
        self.gen_interface(ffct.factory("cs_dfdp_vec",["t","x","adj:out","u","p","p_fixed"],["densify:adj:p"]))


        """

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




        self.initial_conditions = []
        self.final_conditions = []
        """

        self.X = self.opti.variable(*stage.x.shape)
        self.U = self.opti.variable(*stage.u.shape)
        self.P = self.opti.parameter(*stage.p.shape)
        self.t = self.opti.parameter()

        self.raw = [stage.x,stage.u,stage.p,stage.t]
        self.optivar = [self.X, self.U, self.P, self.t]

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

        var_lagrange = []
        var_mayer = []
        obj = MX(stage._objective)
        for e in symvar(obj):
            if "integral" in e.name():
                var_lagrange.append(e)
            else:#elif "at_tf" in e.name():
                var_mayer.append(e)
            #else:
            #    raise Exception("Unknown element in objective: %s" % str(e))

        self.lagrange = substitute([obj], var_mayer, [DM.zeros(e.shape) for e in var_mayer])[0]
        self.mayer = substitute([obj], var_lagrange, [DM.zeros(e.shape) for e in var_lagrange])[0]

    def transcribe_phase2(self, stage, **kwargs):
        
        opti_advanced = self.opti.advanced
        placeholders = kwargs["placeholders"]



        # Total Lagrange integrand
        lagrange = placeholders(self.lagrange,preference=['expose'])
        # Total Mayer term
        mayer = placeholders(self.mayer,preference=['expose'])

        xdes = MX.sym("xdes", stage.x.sparsity())
        udes = MX.sym("udes", stage.u.sparsity())
        lfct = Function("cs_lfct", [stage.t, stage.x, stage.u, self.v, stage.p, xdes, udes], [densify(lagrange)], ["t", "x", "u", "p", "p_fixed", "xdes", "udes"], ["out"])
        self.gen_interface(lfct)
        self.gen_interface(lfct.factory("cs_dldx",["t","x","u","p","p_fixed", "xdes", "udes"],["grad:out:x"]))
        self.gen_interface(lfct.factory("cs_dldu",["t","x","u","p","p_fixed", "xdes", "udes"],["grad:out:u"]))
        self.gen_interface(lfct.factory("cs_dldp",["t","x","u","p","p_fixed", "xdes", "udes"],["grad:out:p"]))

        Vfct = Function("cs_Vfct", [stage.T, stage.x, self.v, stage.p, xdes], [densify(mayer)], ["T", "x", "p", "p_fixed", "xdes"], ["out"])
        self.gen_interface(Vfct)
        self.gen_interface(Vfct.factory("cs_dVdx",["T","x","p","p_fixed","xdes"],["grad:out:x"]))
        self.gen_interface(Vfct.factory("cs_dVdp",["T","x","p","p_fixed","xdes"],["grad:out:p"]))
        self.gen_interface(Vfct.factory("cs_dVdT",["T","x","p","p_fixed","xdes"],["grad:out:T"]))

        eq = [] #
        ineq = [] # <=0
        eq_term = []
        ineq_term = []


        # Process path constraints
        for c, meta, args in stage._constraints["control"]+stage._constraints["integrator"]:
            c = substitute([placeholders(c,preference=['expose'])],self.raw,self.optivar)[0]
            mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
            lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)
            # lb <= canon <= ub
            # Check for infinities
            try:
                lb_inf = np.all(np.array(evalf(lb)==-inf))
            except:
                lb_inf = False
            try:
                ub_inf = np.all(np.array(evalf(ub)==inf))
            except:
                ub_inf = False
            print(mc.type)

            if mc.type == casadi.OPTI_EQUALITY:
                eq.append(canon-ub)
            else:
                assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY, casadi.OPTI_DOUBLE_INEQUALITY]
                if not ub_inf:
                    ineq.append(canon-ub)
                if not lb_inf:
                    ineq.append(lb-canon)

        eq = vvcat(eq)
        ineq = vvcat(ineq)

        gfct = Function("cs_gfct", [stage.t, stage.x, stage.u, self.v, stage.p], [densify(eq)], ["t", "x", "u", "p", "p_fixed"], ["out"])
        self.gen_interface(gfct)
        self.gen_interface(gfct.factory("cs_dgdx_vec",["t", "x", "u", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(gfct.factory("cs_dgdu_vec",["t", "x", "u", "p", "adj:out", "p_fixed"],["densify:adj:u"]))
        self.gen_interface(gfct.factory("cs_dgdp_vec",["t", "x", "u", "p", "adj:out", "p_fixed"],["densify:adj:p"]))

        hfct = Function("cs_hfct", [stage.t, stage.x, stage.u, self.v, stage.p], [densify(ineq)], ["t", "x", "u", "p", "p_fixed"], ["out"])
        self.gen_interface(hfct)
        self.gen_interface(hfct.factory("cs_dhdx_vec",["t", "x", "u", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(hfct.factory("cs_dhdu_vec",["t", "x", "u", "p", "adj:out", "p_fixed"],["densify:adj:u"]))
        self.gen_interface(hfct.factory("cs_dhdp_vec",["t", "x", "u", "p", "adj:out", "p_fixed"],["densify:adj:p"]))


        # Process point constraints
        # Probably should de-duplicate stuff wrt path constraints code
        for c, meta, _ in stage._constraints["point"]:
            # Make sure you resolve u to r_at_t0/r_at_tf
            c = placeholders(c,max_phase=1)
            has_t0 = 'r_at_t0' in [a.name() for a in symvar(c)]
            has_tf = 'r_at_tf' in [a.name() for a in symvar(c)]

            cb = c
            c = substitute([placeholders(c,preference='expose')],self.raw,self.optivar)[0]
            mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
            lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)

            if has_t0:
                # t0
                check = is_linear(canon, stage.x)
                check = check and not depends_on(canon, vertcat(stage.u, self.v))
                assert check, "at t=t0, only equality constraints on x are allowed. Got '%s'" % str(c)
                continue

            # Check for infinities
            try:
                lb_inf = np.all(np.array(evalf(lb)==-inf))
            except:
                lb_inf = False
            try:
                ub_inf = np.all(np.array(evalf(ub)==inf))
            except:
                ub_inf = False

            if mc.type == casadi.OPTI_EQUALITY:
                eq_term.append(canon-ub)
            else:
                assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY, casadi.OPTI_DOUBLE_INEQUALITY]
                if not ub_inf:
                    eq_term.append(canon-ub)
                if not lb_inf:
                    ineq_term.append(lb-canon)

        eq_term = vvcat(eq_term)
        ineq_term = vvcat(ineq_term)

        gTfct = Function("cs_gTfct", [stage.T, stage.x, self.v, stage.p], [densify(eq_term)], ["T", "x", "p", "p_fixed"], ["out"])
        self.gen_interface(gTfct)
        self.gen_interface(gTfct.factory("cs_dgTdx_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(gTfct.factory("cs_dgTdp_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:p"]))
        self.gen_interface(gTfct.factory("cs_dgTdT_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:T"]))

        hTfct = Function("cs_hTfct", [stage.T, stage.x, self.v, stage.p], [densify(ineq_term)], ["T", "x", "p", "p_fixed"], ["out"])
        self.gen_interface(hTfct)
        self.gen_interface(hTfct.factory("cs_dhTdx_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(hTfct.factory("cs_dhTdp_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:p"]))
        self.gen_interface(hTfct.factory("cs_dhTdT_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:T"]))


        X0 = DM.zeros(stage.nx, self.N+1)
        U0 = DM.zeros(stage.nu, self.N)

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

        print("x0, u0")
        X0 = list(np.array(X0).mean(axis=1))
        U0 = list(np.array(U0).mean(axis=1))

        self.output_file.write("""
/** Additional functions required for semi-implicit systems 
    M*dx/dt(t) = f(t0+t,x(t),u(t),p) using the solver RODAS 
    ------------------------------------------------------- **/
/** Jacobian df/dx in vector form (column-wise) **/
void dfdx(typeRNum *out, ctypeRNum t, ctypeRNum *x, ctypeRNum *u, ctypeRNum *p, typeUSERPARAM *userparam)
{
}
/** Jacobian df/dx in vector form (column-wise) **/
void dfdxtrans(typeRNum *out, ctypeRNum t, ctypeRNum *x, ctypeRNum *u, ctypeRNum *p, typeUSERPARAM *userparam)
{
}
/** Jacobian df/dt **/
void dfdt(typeRNum *out, ctypeRNum t, ctypeRNum *x, ctypeRNum *u, ctypeRNum *p, typeUSERPARAM *userparam)
{
}
/** Jacobian d(dH/dx)/dt  **/
void dHdxdt(typeRNum *out, ctypeRNum t, ctypeRNum *x, ctypeRNum *u, ctypeRNum *vec, ctypeRNum *p, typeUSERPARAM *userparam)
{
}
/** Mass matrix in vector form (column-wise, either banded or full matrix) **/
void Mfct(typeRNum *out, typeUSERPARAM *userparam)
{
}
/** Transposed mass matrix in vector form (column-wise, either banded or full matrix) **/
void Mtrans(typeRNum *out, typeUSERPARAM *userparam)
{
}
        """)
        self.output_file.write(f"""
            /** OCP dimensions: states (Nx), controls (Nu), parameters (Np), equalities (Ng), 
            inequalities (Nh), terminal equalities (NgT), terminal inequalities (NhT) **/
            void ocp_dim(typeInt *Nx, typeInt *Nu, typeInt *Np, typeInt *Ng, typeInt *Nh, typeInt *NgT, typeInt *NhT, typeUSERPARAM *userparam)
            {{
                *Nx = {stage.nx};
                *Nu = {stage.nu};
                *Np = {self.v.numel()};
                *Ng = {eq.numel()};
                *Nh = {ineq.numel()};
                *NgT = {eq_term.numel()};
                *NhT = {ineq_term.numel()};
            }}
        """)


        self.output_file.write("void preamble(typeUSERPARAM* userparam) {\n")
        for l in self.preamble:
            self.output_file.write("  " + l + "\n")
        self.output_file.write(f"""
        {self.user}->arg = malloc(sizeof(const casadi_real*)*sz_arg);
        {self.user}->res = malloc(sizeof(casadi_real*)*sz_res);
        {self.user}->iw = sz_iw>0 ? malloc(sizeof(casadi_int)*sz_iw) : 0;
        {self.user}->w = sz_w>0 ? malloc(sizeof(casadi_real)*sz_w) : 0;


        """)
        self.output_file.write("}\n")

        self.output_file.write("void postamble(typeUSERPARAM* userparam) {\n")
        for l in self.postamble:
            self.output_file.write("  " + l + "\n")
        self.output_file.write(f"  free({self.user}->arg);\n")
        self.output_file.write(f"  free({self.user}->res);\n")
        self.output_file.write(f"  free({self.user}->iw);\n")
        self.output_file.write(f"  free({self.user}->w);\n")
        self.output_file.write("}\n")   

        nc = vertcat(eq,ineq,eq_term,ineq_term).numel()
        self.output_file.write("typeGRAMPC* setup() {\n")
        print(self.control_grid_init)
        if not isinstance(self.ConstraintsAbsTol,list):
            self.ConstraintsAbsTol = [self.ConstraintsAbsTol] * nc
        self.output_file.write(f"""
            typeGRAMPC *grampc;
            typeUSERPARAM* userparam = malloc(sizeof(cs_struct));
            double ConstraintsAbsTol[{nc}] = {{{",".join("%.16e" % e for e in self.ConstraintsAbsTol)}}};
            double x0[{stage.nx}] = {{{strlist(X0)}}};
            double u0[{stage.nu}] = {{{strlist(U0)}}};

            preamble(userparam);

            /********* grampc init *********/
            grampc_init(&grampc, userparam);

            grampc_setparam_real_vector(grampc, "x0", x0);
            //grampc_setparam_real_vector(grampc, "xdes", 0);

            grampc_setparam_real_vector(grampc, "u0", u0);
            //grampc_setparam_real_vector(grampc, "udes", 0);
            //grampc_setparam_real_vector(grampc, "umax", umax);
            //grampc_setparam_real_vector(grampc, "umin", umin);

            grampc_setparam_real(grampc, "Thor", {self.control_grid_init[-1]-self.control_grid_init[0]});

            grampc_setparam_real(grampc, "dt", {self.control_grid_init[1]-self.control_grid_init[0]});
            grampc_setparam_real(grampc, "t0", {self.control_grid_init[0]});

            /********* Option definition *********/
            grampc_setopt_int(grampc, "Nhor", {self.Nhor});
            grampc_setopt_int(grampc, "MaxGradIter", {self.MaxGradIter});
            grampc_setopt_int(grampc, "MaxMultIter", {self.MaxMultIter});
            grampc_setopt_string(grampc, "ShiftControl", "{self.ShiftControl}");

            grampc_setopt_real(grampc, "LineSearchMax", {self.LineSearchMax});
            grampc_setopt_string(grampc, "LineSearchExpAutoFallback", "{self.LineSearchExpAutoFallback}");

            grampc_setopt_real(grampc, "AugLagUpdateGradientRelTol", {self.AugLagUpdateGradientRelTol});
            grampc_setopt_real_vector(grampc, "ConstraintsAbsTol", ConstraintsAbsTol);

            grampc_setopt_real(grampc, "PenaltyMax", {self.PenaltyMax});
            grampc_setopt_real(grampc, "PenaltyMin", {self.PenaltyMin});
            grampc_setopt_real(grampc, "PenaltyIncreaseFactor", {self.PenaltyIncreaseFactor});
            grampc_setopt_real(grampc, "PenaltyDecreaseFactor", {self.PenaltyDecreaseFactor});

            grampc_setopt_string(grampc, "ConvergenceCheck", "{self.ConvergenceCheck}");
            grampc_setopt_real(grampc, "ConvergenceGradientRelTol", {self.ConvergenceGradientRelTol});

            /********* estimate and set PenaltyMin *********/
            grampc_estim_penmin(grampc, 1);

            grampc_printopt(grampc);
            grampc_printparam(grampc);

            return grampc;
        """)
        self.output_file.write("}\n")

        self.output_file.write("void solve(typeGRAMPC* grampc) {\n")
        self.output_file.write(f"""
            /* run grampc */
            printf("Running GRAMPC!\\n");
            grampc_run(grampc);
            grampc_printstatus(grampc->sol->status, STATUS_LEVEL_DEBUG);

            /* run convergence test */
            /*if (grampc->opt->ConvergenceCheck == INT_ON) {{
                converged_grad = convergence_test_gradient(grampc->opt->ConvergenceGradientRelTol, grampc);
                if (converged_grad) {{
                    converged_const = convergence_test_constraints(grampc->opt->ConstraintsAbsTol, grampc);
                }}
            }}*/

        """)
        self.output_file.write("}\n")


        self.output_file.write("void destroy(typeGRAMPC* grampc) {\n")
        self.output_file.write(f"""
            postamble(grampc->userparam);
            free(grampc->userparam);

        """)
        self.output_file.write("}\n")

        self.output_file.write("int main() {\n")
        self.output_file.write(f"""
            typeGRAMPC* s = setup();
            solve(s);
            destroy(s);
        """)
        self.output_file.write("}\n")
        
     

        self.output_file.close()
        self.codegen.generate()
        

        build_dir_abs = "."

        cmake_file_name = os.path.join(build_dir_abs,"CMakeLists.txt")
        with open(cmake_file_name,"w") as out:
            out.write(f"""
            project(grampc_export)

            set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

            cmake_minimum_required(VERSION 3.0)

            add_library(grampc INTERFACE)

            find_library(GRAMPC_LIB NAMES grampc HINTS /home/jgillis/programs/grampc-code/libs)
            find_path(GRAMPC_INCLUDE_DIR
            grampc.h
            HINTS /home/jgillis/programs/grampc-code/include
            )

            target_link_libraries(grampc INTERFACE ${{GRAMPC_LIB}} m)
            target_include_directories(grampc INTERFACE ${{GRAMPC_INCLUDE_DIR}})

            add_library({self.grampc_driver} SHARED {self.grampc_driver}.c {self.codegen_name}.c)
            add_executable({self.grampc_driver}_main {self.grampc_driver}.c {self.codegen_name}.c)
            
            target_link_libraries({self.grampc_driver} grampc)
            target_link_libraries({self.grampc_driver}_main grampc)

            install(TARGETS {self.grampc_driver} {self.grampc_driver}_main RUNTIME DESTINATION . LIBRARY DESTINATION .)

            """)
        cmake_file_name = os.path.join(build_dir_abs,"build.bat")
        with open(cmake_file_name,"w") as out:
            out.write(f"""
            echo "Should be ran in 'x64 Native Tools Command Prompt for VS'"
            cmake -G "Visual Studio 16 2019" -A x64 -B build
            cmake --build build --config Release
            cmake --install build --prefix .
            """)
        cmake_file_name = os.path.join(build_dir_abs,"build.sh")
        with open(cmake_file_name,"w") as out:
            out.write(f"""
            cmake -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE=Debug -B build
            cmake --build build
            cmake --install build --prefix .
            """)
        subprocess.run(["bash","build.sh"])
            
        # PyDLL instead of CDLL to keep GIL:
        # virtual machine emits Python prints
        if os.name == "nt":
            libname = self.grampc_driver+".dll"
        else:
            libname = "lib"+self.grampc_driver+".so"
        self.lib = PyDLL(os.path.join(build_dir_abs,libname))

        # Type aliases
        m_type = c_void_p
        CONST = lambda x: x

        self._register("setup",[], m_type)
        self._register("destroy",[m_type], m_type)
        self._register("solve",[m_type], m_type)

        grampc = self._setup()
        print(grampc)
        self._solve(grampc)
        self._destroy(grampc)


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