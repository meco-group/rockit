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
from casadi import vec, CodeGenerator, SX, Sparsity, MX, vcat, veccat, symvar, substitute, densify, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag
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


TODO: debug with example from repo, make callbacks print

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

class GrampcMethod(ExternalMethod):
    def __init__(self,
    	M = 1,
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
        self.M = M
        self.codegen_name = 'casadi_codegen'
        self.grampc_driver = 'grampc_driver'
        self.user = "((cs_struct*) userparam)"
        self.user_grampc = "((cs_struct*) grampc->userparam)"
        self.Nhor = self.N*self.M
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
        adj_i = None
        for i in range(f.n_in()):
            e = f.name_in(i)
            if "adj" in e:
                adj_i = i
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
        self.output_file.write(f"  printf(\"{f.name()[3:]}: \");\n")
        if adj_i is not None:
            self.output_file.write(f"  for (int i=0;i<{f.numel_in(adj_i)};++i) printf(\"%e \", {f.name_in(adj_i)}[i]);\n")
            self.output_file.write(f"  printf(\" -> \");\n")
        self.output_file.write(f"  for (int i=0;i<{f.numel_out(0)};++i) printf(\"%e \", out[i]);\n")
        self.output_file.write(f"  printf(\"\\n\");\n")
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
            casadi_real* x_opt;
            casadi_real* u_opt;
            casadi_real* v_opt;
            casadi_real* x_current;
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
        self.P0 = DM.zeros(stage.np)

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

        print(ineq)

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

        # No export_num here, let's do things parametrically
        self.m = m = OrderedDict()

        x0_eq = []
        x0_b = []

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
                assert check and mc.type == casadi.OPTI_EQUALITY, "at t=t0, only equality constraints on x are allowed. Got '%s'" % str(c)

                J,c = linear_coeff(canon, stage.x)
                J = evalf(J)
                x0_eq.append(J)
                x0_b.append(lb-c)

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

        x0_eq = vcat(x0_eq)
        x0_b = vcat(x0_b)
        x0_expr = casadi.solve(x0_eq, x0_b)
        m["x_current"] = export_vec(x0_expr)

        eq_term = vvcat(eq_term)
        ineq_term = vvcat(ineq_term)

        gTfct = Function("cs_gTfct", [stage.T, stage.x, self.v, stage.p], [densify(eq_term)], ["T", "x", "p", "p_fixed"], ["out"])
        self.gen_interface(gTfct)
        self.gen_interface(gTfct.factory("cs_dgTdx_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(gTfct.factory("cs_dgTdp_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:p"]))
        self.gen_interface(gTfct.factory("cs_dgTdT_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:T"]))

        print(ineq_term)
        hTfct = Function("cs_hTfct", [stage.T, stage.x, self.v, stage.p], [densify(ineq_term)], ["T", "x", "p", "p_fixed"], ["out"])
        self.gen_interface(hTfct)
        self.gen_interface(hTfct.factory("cs_dhTdx_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(hTfct.factory("cs_dhTdp_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:p"]))
        self.gen_interface(hTfct.factory("cs_dhTdT_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:T"]))

        args = [v[0] for v in m.values()]
        self.mmap = Function('mmap',[stage.p],args,['p'],list(m.keys()))

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
        {self.user}->x_opt = malloc(sizeof(casadi_real)*{stage.nx*self.Nhor});
        {self.user}->u_opt = malloc(sizeof(casadi_real)*{stage.nu*self.Nhor});
        //{self.user}->v_opt = malloc(sizeof(casadi_real)*{self.v.numel()});
        {self.user}->x_current = malloc(sizeof(casadi_real)*{stage.nx});

        """)
        self.output_file.write("}\n")

        self.output_file.write("void postamble(typeUSERPARAM* userparam) {\n")
        for l in self.postamble:
            self.output_file.write("  " + l + "\n")
        self.output_file.write(f"  free({self.user}->arg);\n")
        self.output_file.write(f"  free({self.user}->res);\n")
        self.output_file.write(f"  free({self.user}->iw);\n")
        self.output_file.write(f"  free({self.user}->w);\n")
        self.output_file.write(f"  free({self.user}->x_opt);\n")
        self.output_file.write(f"  free({self.user}->u_opt);\n")
        #self.output_file.write(f"  free({self.user}->v_opt);\n")
        self.output_file.write(f"  free({self.user}->x_current);\n")
        self.output_file.write("}\n")   

        nc = vertcat(eq,ineq,eq_term,ineq_term).numel()
        self.output_file.write("typeGRAMPC* setup() {\n")
        if not isinstance(self.ConstraintsAbsTol,list):
            self.ConstraintsAbsTol = [self.ConstraintsAbsTol] * nc
        self.output_file.write(f"""
            typeGRAMPC *grampc;
            typeUSERPARAM* userparam = malloc(sizeof(cs_struct));
            double ConstraintsAbsTol[{nc}] = {{{",".join("%.16e" % e for e in self.ConstraintsAbsTol)}}};
            //double x0[{stage.nx}] = 
            //double u0[{stage.nu}] = 

            preamble(userparam);

            /********* grampc init *********/
            grampc_init(&grampc, userparam);

            //grampc_setparam_real_vector(grampc, "x0", x0);
            //grampc_setparam_real_vector(grampc, "xdes", 0);

            //grampc_setparam_real_vector(grampc, "u0", u0);
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

        self.output_file.write(f"""
            void write_p(typeGRAMPC* grampc, const casadi_real* p) {{
                for (int i=0;i<{stage.np};++i) {self.user_grampc}->p[i] = p[i];
            }}
            void write_x_current(typeGRAMPC* grampc, const casadi_real* x_current) {{
                for (int i=0;i<{stage.nx};++i) {self.user_grampc}->x_current[i] = x_current[i];
            }}
            void read_x_opt(typeGRAMPC* grampc, casadi_real* x_opt) {{
                for (int i=0;i<{stage.nx*self.Nhor};++i) x_opt[i] = {self.user_grampc}->x_opt[i];
            }}
            void read_u_opt(typeGRAMPC* grampc, casadi_real* u_opt) {{
                for (int i=0;i<{stage.nu*self.Nhor};++i) u_opt[i] = {self.user_grampc}->u_opt[i];
            }}""")

        self.output_file.write("void solve(typeGRAMPC* grampc) {\n")
        self.output_file.write(f"""
            /* run grampc */
            printf("Running GRAMPC!\\n");
            grampc_setparam_real_vector(grampc, "x0", {self.user_grampc}->x_current);

            grampc_run(grampc);
            grampc_printstatus(grampc->sol->status, STATUS_LEVEL_DEBUG);

            /* run convergence test */
            /*if (grampc->opt->ConvergenceCheck == INT_ON) {{
                converged_grad = convergence_test_gradient(grampc->opt->ConvergenceGradientRelTol, grampc);
                if (converged_grad) {{
                    converged_const = convergence_test_constraints(grampc->opt->ConstraintsAbsTol, grampc);
                }}
            }}*/


            for(int k=0;k<grampc->opt->Nhor;++k) {{
                for (int j=0;j<{stage.nx};++j) {{
                    int i = k*{stage.nx}+j;
                    {self.user_grampc}->x_opt[i] = grampc->rws->x[i]*grampc->opt->xScale[j]+grampc->opt->xOffset[j];
                }}
                for (int j=0;j<{stage.nu};++j) {{
                    int i = k*{stage.nu}+j;
                    {self.user_grampc}->u_opt[i] = grampc->rws->u[i]*grampc->opt->uScale[j]+grampc->opt->uOffset[j];
                }}
            }}




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
        self._register("read_x_opt",[m_type, POINTER(c_double)], m_type)
        self._register("read_u_opt",[m_type, POINTER(c_double)], m_type)
        self._register("write_p",[m_type, POINTER(c_double)], m_type)
        self._register("write_x_current",[m_type, POINTER(c_double)], m_type)
        self.grampc = self._setup()

    def set_matrices(self):
        P0 = np.array(self.P0)

        res = self.mmap(p=self.P0)

        self._write_p(self.grampc, P0.ctypes.data_as(POINTER(c_double)))
        x_current = np.array(res["x_current"])
        self._write_x_current(self.grampc, x_current.ctypes.data_as(POINTER(c_double)))

    def __del__(self):
        if hasattr(self,'_destroy'):
            self._destroy(self.grampc)

    def solve(self, stage):
        self.set_matrices()
        self._solve(self.grampc)
        x_opt = np.zeros((stage.nx, self.Nhor+1),dtype=np.float64,order='F')
        self._read_x_opt(self.grampc, x_opt.ctypes.data_as(POINTER(c_double)))
        u_opt = np.zeros((stage.nu, self.Nhor),dtype=np.float64,order='F')
        self._read_u_opt(self.grampc, u_opt.ctypes.data_as(POINTER(c_double)))
        return OcpSolution(SolWrapper(self, vec(x_opt), vec(u_opt)), stage)

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