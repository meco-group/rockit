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
from ...casadi_helpers import prepare_build_dir
from ..method import ExternalMethod, linear_coeffs, check_Js
from ...solution import OcpSolution

from .code_generator import GrampcCodeGenerator

import numpy as np
from casadi import external, vec, CodeGenerator, SX, Sparsity, MX, vcat, veccat, symvar, substitute, densify, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag, solve, fmin, fmax
import casadi
import casadi as cs
from ...casadi_helpers import DM2numpy, reshape_number
from collections import OrderedDict

import subprocess
import os
from ctypes import *
import glob

INF = 1e20
"""
In general, the constraints should be formulated in such a way that there are no conflicts. However,
numerical difficulties can arise in some problems if constraints are formulated twice for the last point.
Therefore, GRAMPC does not evaluate the constraints g and h for the last trajectory point if terminal
constraints are defined, i.e. NgT + NhT > 0. In contrast, if no terminal constraints are defined, the
functions g and h are evaluated for all points. Note that the opposite behavior is easy to implement
by including g and h in the terminal constraints gT and hT .


TODO: debug with example from repo, make callbacks print

The convergence checks are a bit wird. The gradient of the Lagrangian is not checked; a small step size in decision space is enough

GRAMPC is an indirect approach (first optimize then discretize).
The boundary value problem is discretised using single shooting (we integrate over the entire time horizon).
The inner part of the min-max problem is solved using a projected gradient method (first order).lsAdapt


Debugging test1.py
Suspected that convergence ends prematurely due to poorly implemented linesearch.
Played around with linesearch options, also experimented with resetting the adaptive strategy after alpha shrinks a lot -> no difference

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

class MyCodeGenerator:
    def __init__(self,name):
        self.added_shorthands = set()
        self.add_includes = []
        self.prefix = ""
        self.auxiliaries = ""
        self.body = ""
        self.name = name

    def add_include(self,h):
        self.add_includes.append(h)

    def shorthand(self,name):
        self.added_shorthands.add(name)
        return self.prefix + name

    def add_dependency(self, f):
        name = f.codegen_name(self, False)
        fname = self.prefix + name
        stack_counter = self.shorthand(name + "_unused_stack_counter")
        stack = self.shorthand(name + "_unused_stack")
        mem_counter = self.shorthand(name + "_mem_counter")
        mem_array = self.shorthand(name + "_mem")
        alloc_mem = self.shorthand(name + "_alloc_mem")
        init_mem = self.shorthand(name + "_init_mem")
        work = self.shorthand(name+"_work")

        self.auxiliaries += f"static int {mem_counter} = 0;\n"
        self.auxiliaries += f"static int {stack_counter } = -1;\n"
        self.auxiliaries += f"static int {stack}[CASADI_MAX_NUM_THREADS];\n"
        self.auxiliaries += f"static {f.codegen_mem_type()} *{mem_array}[CASADI_MAX_NUM_THREADS];\n\n"

        f.codegen_declarations(self)

        f.codegen(self, fname)

        def encode_sp(sp,i):
            r = f"case {i}:\n"
            r+= "{"
            spc = sp.compress()
            r+= f"static casadi_int sp[{len(spc)}] = {{{strlist(spc)}}}; return sp;"
            r+= "}"
            return r

        def encode_name(n,i):
            return f"case {i}: return \"{n}\";\n"

        newline = "\n"

        self.body += f"""

            casadi_int {fname}_n_in(void) {{
                return {f.n_in()};
            }}

            casadi_int {fname}_n_out(void) {{
                return {f.n_out()};
            }}

            const casadi_int* {fname}_sparsity_in(casadi_int i) {{
                switch (i) {{
                {newline.join(encode_sp(f.sparsity_in(i), i) for i in range(f.n_in()))}
                default: return 0;
                }}
            }}

            const casadi_int* {fname}_sparsity_out(casadi_int i) {{
                switch (i) {{
                {newline.join(encode_sp(f.sparsity_out(i), i) for i in range(f.n_out()))}
                default: return 0;
                }}
            }}

            const char* {fname}_name_in(casadi_int i) {{
                switch (i) {{
                {newline.join(encode_name(f.name_in(i), i) for i in range(f.n_in()))}
                default: return 0;
                }}
            }}

            const char* {fname}_name_out(casadi_int i) {{
                switch (i) {{
                {newline.join(encode_name(f.name_out(i), i) for i in range(f.n_out()))}
                default: return 0;
                }}
            }}

            int {fname}_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {{
            {f.codegen_work(self)}
              return 0;
            }}

            // Alloc memory
            int {fname}_alloc_mem(void) {{
            {f.codegen_alloc_mem(self)}
            }}

            // Initialize memory
            int {fname}_init_mem(int mem) {{
            {f.codegen_init_mem(self)}
            }}

            // Clear memory
            int {fname}_free_mem(int mem) {{
            {f.codegen_free_mem(self)}
            }}

            int {self.shorthand(name + "_checkout")}(void) {{
            int mid;
            if ({stack_counter}>=0) {{
              return {stack}[{stack_counter}--];
            }} else {{
              if ({mem_counter}==CASADI_MAX_NUM_THREADS) return -1;
              mid = {alloc_mem}();
              if (mid<0) return -1;
              if({init_mem}(mid)) return -1;
              return mid;
            }}

            return {stack}[{stack_counter}--];
        }}

        void {self.shorthand(name+"_release")}(int mem) {{
            {stack}[++{stack_counter}] = mem;
        }}


        """

    def generate(self,dir="."):
        with open(os.path.join(dir,self.name+".c"),"w") as out:
            out.write("#define CASADI_MAX_NUM_THREADS 1\n")
            for e in self.add_includes:
                out.write(f"#include \"{e}\"\n")
            out.write(self.auxiliaries)
            out.write(self.body)

class Wrapper:
    def __init__(self,userparam):
        self.added_declarations = ""
        self.added_init_mem = ""
        self.added_body = ""
        self.sp_in = []
        self.sp_out = []
        self._name_in = []
        self._name_out = []
        self.userparam = userparam

    def set_sp_in(self, sp):
        self.sp_in = sp

    def set_sp_out(self, sp):
        self.sp_out = sp

    def set_name_in(self, name):
        self._name_in = name

    def set_name_out(self, name):
        self._name_out = name

    def add_declarations(self, decl):
        self.added_declarations += decl
    
    def add_init_mem(self, init_mem):
        self.added_init_mem += init_mem

    def add_body(self, body):
        self.added_body += body

    def signature(self, fname):
        return "int " + fname + "(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem)"

    def codegen(self, g, fname):
        g.body += f"{self.signature(fname)} {{\n"
        self.codegen_body(g)
        g.body += f"return 0;}}\n"

    def codegen_name(self, g, ns):
        return "grampc_driver"
    def codegen_mem_type(self):
        return "typeGRAMPC"
    def codegen_alloc_mem(self, g):
        name = self.codegen_name(g, False)
        mem_counter = g.shorthand(name + "_mem_counter")
        return f"return {mem_counter}++;"

    def codegen_init_mem(self, g):
        grampc = self.codegen_mem(g, "mem")
        userparam = grampc+"->userparam"
        return f"""
        typeGRAMPC *grampc = {grampc};
        {self.added_init_mem}
        {grampc} = grampc;
        return 0;
        
        """

    def codegen_free_mem(self, g):
        grampc = self.codegen_mem(g)
        userparam = grampc+"->userparam"
        return f"postamble({userparam});\nfree({userparam});"

    def codegen_mem(self, g, index=0):
        name = self.codegen_name(g, False)
        mem_array = g.shorthand(name + "_mem")
        return mem_array+"[" + str(index) + "]"

    def codegen_body(self, g):
        grampc = self.codegen_mem(g, "mem")
        g.add_include("grampc.h")
        g.body += f"typeGRAMPC *grampc = {grampc};"
        stats = f"{self.userparam}->stats"
        g.body += f"{stats}.stop_crit=0;"
        g.body += f"{stats}.n_outer_iter=0;"
        g.body += f"{stats}.n_inner_iter=0;"
        g.body += f"{stats}.runtime=0;"
        g.body += self.added_body

    def n_in(self):
        return len(self.sp_in)

    def n_out(self):
        return len(self.sp_out)

    def sparsity_in(self, i):
        return self.sp_in[i]

    def sparsity_out(self, i):
        return self.sp_out[i]

    def name_in(self, i):
        return self._name_in[i]

    def name_out(self, i):
        return self._name_out[i]


    def codegen_work(self, g):
        r = f"""
            casadi_int sz_arg_local, sz_res_local, sz_iw_local, sz_w_local;
            *sz_arg=0, *sz_res=0, *sz_iw=0, *sz_w=0;
            pmap_work(&sz_arg_local, &sz_res_local, &sz_iw_local, &sz_w_local);
            if (sz_arg_local>*sz_arg) *sz_arg=sz_arg_local;
            if (sz_res_local>*sz_res) *sz_res=sz_res_local;
            if (sz_iw_local>*sz_iw) *sz_iw=sz_iw_local;
            if (sz_w_local>*sz_iw) *sz_w=sz_w_local;

            *sz_arg += {self.n_in()};
            *sz_res += {self.n_out()};
        """
        return r


    def codegen_declarations(self, g):
        g.auxiliaries += """


        typedef struct {
            int stop_crit;
            int conv_grad;
            int conv_con;
            int n_outer_iter;
            int n_inner_iter;
            double runtime;
        } grampc_solver_stats;

        typedef struct {
            int sqp_stop_crit;
            int n_sqp_iter;
            int n_ls;
            int n_max_ls;
            int n_qp_iter;
            int n_max_qp_iter;
            double runtime;
        } compat_solver_stats;

        typedef struct cs_struct_def {
            const casadi_real** arg;
            casadi_real** res;
            casadi_int* iw;
            casadi_real* w;
            casadi_real* p;
            typeGRAMPC* grampc;
            casadi_real* x_opt;
            casadi_real* u_opt;
            casadi_real T_opt;
            casadi_real* v_opt;
            casadi_real* x_current;
            casadi_real* umin;
            casadi_real* umax;
            casadi_real* u0;
            casadi_real* v0;
            casadi_real Tmin;
            casadi_real Tmax;
            grampc_solver_stats stats;
        } cs_struct;

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

        """
        g.auxiliaries += self.added_declarations


class SourceArtifact:
    def __init__(self, name):
        self.name = name

class GrampcMethod(ExternalMethod):
    def __init__(self,
        verbose=True,
        grampc_options=None,
        **kwargs):
        """
        dt is post-processing: interplin
        

        GRAMPC is very much realtime iteration
        By default, ConvergenceCheck is even off, and you perform MaxMultIter outer iterations with MaxGradIter inner iterations.
        Default MaxMultIter = 1.

        
        """
        supported = {"free_T"}
        ExternalMethod.__init__(self, supported=supported, **kwargs)
        self.grampc_options = {} if grampc_options is None else grampc_options
        our_defaults = {"MaxMultIter": 3000, "ConvergenceCheck": "on", "MaxGradIter": 100, "ConstraintsAbsTol": 1e-8, "ConvergenceGradientRelTol": 1e-8,"LineSearchType": "adaptive"}
        for k,v in our_defaults.items():
            if k not in self.grampc_options:
                self.grampc_options[k] = v
        self.codegen_name = 'casadi_codegen'
        self.grampc_driver = 'grampc_driver'
        self.user = "((cs_struct*) userparam)"
        self.user_grampc = "((cs_struct*) grampc->userparam)"
        self.Nhor = self.N+1
        self.verbose = verbose
        self.artifacts = []

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
        self.wrapper.add_declarations(f"void {f.name()[3:]}(typeRNum *out, {', '.join(args)}, typeUSERPARAM *userparam) {{\n")
        self.wrapper.add_declarations("  int mem;\n")
        adj_i = None
        for i in range(f.n_in()):
            e = f.name_in(i)
            if "adj" in e:
                adj_i = i
            if scalar(e):
                self.wrapper.add_declarations(f"  {self.user}->arg[{i}] = &{e};\n")
            elif e=="p_fixed":
                self.wrapper.add_declarations(f"  {self.user}->arg[{i}] = {self.user}->p;\n")
            else:
                self.wrapper.add_declarations(f"  {self.user}->arg[{i}] = {e};\n")
        self.wrapper.add_declarations(f"  {self.user}->res[0] = out;\n")
        self.wrapper.add_declarations(f"  mem = {f.name()}_checkout();\n")
        self.wrapper.add_declarations(f"  {f.name()}({self.user}->arg, {self.user}->res, {self.user}->iw, {self.user}->w, mem);\n")
        self.wrapper.add_declarations(f"  {f.name()}_release(mem);\n")
        self.wrapper.add_declarations("}\n")

    def transcribe_phase1(self, stage, **kwargs):

        self.preamble = ["casadi_int sz_arg=0, sz_res=0, sz_iw=0, sz_w=0;",
                         "casadi_int sz_arg_local, sz_res_local, sz_iw_local, sz_w_local;",
                        ]
        self.postamble = []

        self.stage = stage
        self.opti = Opti()



        f = stage._ode()
        options = {}
        options["with_header"] = True
        self.codegen = CodeGenerator(f"{self.codegen_name}.c", options)

        self.artifacts.append(SourceArtifact(f"{self.codegen_name}.c"))
        self.artifacts.append(SourceArtifact(f"{self.codegen_name}.h"))

        self.wrapper_codegen = MyCodeGenerator(self.grampc_driver)
        self.wrapper_codegen.add_include(f"{self.codegen_name}.h")
        self.artifacts.append(SourceArtifact(f"{self.grampc_driver}.c"))
        self.wrapper = Wrapper(userparam=self.user_grampc)

        assert len(stage.variables['control'])==0, "variables defined on control grid not supported. Use controls instead."

        self.v = vvcat(stage.variables[''])
        self.X_gist = [MX.sym("Xg", stage.nx) for k in range(self.N+1)]
        self.U_gist = [MX.sym("Ug", stage.nu) for k in range(self.N)]
        self.V_gist = MX.sym("Vg", *self.v.shape)
        self.T_gist = MX.sym("Tg")

        assert f.numel_out("alg")==0
        assert f.numel_out("quad")==0
        ffct = Function("cs_ffct", [stage.t, stage.x, stage.u, self.v, stage.p], [ densify(f(x=stage.x, u=stage.u, p=stage.p, t=stage.t)["ode"])],['t','x','u','p','p_fixed'],['out'])
        self.gen_interface(ffct)
        self.gen_interface(ffct.factory("cs_dfdx_vec",["t","x","adj:out","u","p","p_fixed"],["densify:adj:x"]))
        self.gen_interface(ffct.factory("cs_dfdu_vec",["t","x","adj:out","u","p","p_fixed"],["densify:adj:u"]))
        self.gen_interface(ffct.factory("cs_dfdp_vec",["t","x","adj:out","u","p","p_fixed"],["densify:adj:p"]))

        self.X = self.opti.variable(*stage.x.shape)
        self.U = self.opti.variable(*stage.u.shape)
        self.V = self.opti.variable(*self.v.shape)
        self.P = self.opti.parameter(*stage.p.shape)
        self.t = self.opti.parameter()
        self.T = self.opti.variable()

        self.raw = [stage.x,stage.u,stage.p,stage.t, self.v]
        self.optivar = [self.X, self.U, self.P, self.t, self.V]
        if self.free_time:
            self.raw += [stage.T]
            self.optivar += [self.T]

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
            elif "at_tf" in e.name():
                var_mayer.append(e)
            else:
                raise Exception("Unknown element in objective: %s" % str(e))

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

        # helpers to put limits on u
        ub_expr = []
        ub_l = []
        ub_u = []



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

                # Catch simple bounds on u
                if is_linear(canon, stage.u) and not depends_on(canon, vertcat(stage.x, self.v)):
                    J,c = linear_coeff(canon, stage.u)
                    try:
                        check_Js(J)
                        ub_expr.append(J)
                        if ub_inf:
                            ub_u.append(reshape_number(J @ stage.u,-INF))
                        else:
                            ub_u.append(ub-c)
                        if lb_inf:
                            ub_l.append(reshape_number(J @ stage.u,INF))
                        else:
                            ub_l.append(lb-c)
                        continue
                    except:
                        pass

                if not ub_inf:
                    ineq.append(canon-ub)
                if not lb_inf:
                    ineq.append(lb-canon)

        eq = vvcat(eq)
        ineq = vvcat(ineq)

        ub_expr = evalf(vcat(ub_expr))
        ub_l = evalf(vcat(ub_l))
        ub_u = evalf(vcat(ub_u))
        # Add missing rows
        rows = set(sum1(ub_expr).T.row())
        missing_rows = [i for i in range(stage.nu) if i not in rows]
        M = DM(len(missing_rows), stage.nu)
        for i,e in enumerate(missing_rows):
            M[i,e] = 1

        ub_expr = vertcat(ub_expr,M)
        ub_l = vertcat(ub_l,-INF*DM.ones(len(missing_rows)))
        ub_u = vertcat(ub_u,INF*DM.ones(len(missing_rows)))
       
        ub_l = solve(ub_expr,ub_l)
        ub_u = solve(ub_expr,ub_u)

        # No export_num here, let's do things parametrically
        self.m = m = OrderedDict()

        m["umin"] = export_vec(ub_l)
        m["umax"] = export_vec(ub_u)

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


        x0_eq = []
        x0_b = []

        Tmin = 0
        Tmax = INF

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
                try:
                    J = evalf(J)
                    x0_eq.append(J)
                    x0_b.append(lb-c)
                    continue
                except:
                    pass

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
                # Catch simple bounds on T
                if self.free_time:
                    if is_linear(canon, stage.T) and not depends_on(canon, vertcat(stage.x, stage.u, self.v)):
                        J,c = linear_coeff(canon, stage.T)
                        if not ub_inf:
                            Tmax = fmin(Tmax, (ub-c)/J)
                        if not lb_inf:
                            Tmin = fmax(Tmin, (lb-c)/J)
                        continue

                if not ub_inf:
                    ineq_term.append(canon-ub)
                if not lb_inf:
                    ineq_term.append(lb-canon)

        Tmin = fmax(Tmin, 1/INF)
        m["Tmin"] = export(Tmin)
        m["Tmax"] = export(Tmax)

        x0_eq = vcat(x0_eq)
        x0_b = vcat(x0_b)
        x0_expr = casadi.inv(evalf(x0_eq)) @ x0_b # casadi.solve(x0_eq, x0_b)
        m["x_current"] = export_vec(x0_expr)

        eq_term = vvcat(eq_term)
        ineq_term = vvcat(ineq_term)

        gTfct = Function("cs_gTfct", [stage.T, stage.x, self.v, stage.p], [densify(eq_term)], ["T", "x", "p", "p_fixed"], ["out"])
        gTfct.disp(True)
        self.gen_interface(gTfct)
        self.gen_interface(gTfct.factory("cs_dgTdx_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(gTfct.factory("cs_dgTdp_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:p"]))
        self.gen_interface(gTfct.factory("cs_dgTdT_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:T"]))

        hTfct = Function("cs_hTfct", [stage.T, stage.x, self.v, stage.p], [densify(ineq_term)], ["T", "x", "p", "p_fixed"], ["out"])
        hTfct.disp(True)
        self.gen_interface(hTfct)
        self.gen_interface(hTfct.factory("cs_dhTdx_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:x"]))
        self.gen_interface(hTfct.factory("cs_dhTdp_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:p"]))
        self.gen_interface(hTfct.factory("cs_dhTdT_vec",["T", "x", "p", "adj:out", "p_fixed"],["densify:adj:T"]))

        args = [v[0] for v in m.values()]
        self.pmap = Function('pmap',[stage.p],args,['p'],list(m.keys()))
        self.codegen.add(self.pmap)

        self.U0 = U0 = DM.zeros(stage.nu)
        self.V0 = V0 = DM.zeros(self.v.numel())


        for var, expr in stage._initial.items():
            assert not depends_on(expr, stage.t)
            assert not depends_on(expr, stage.x)
            if depends_on(var,stage.u):
                assert not depends_on(var, self.v)
                J, r = linear_coeffs(var,stage.u)
                J = evalf(J)
                r = evalf(r)
                assert r.is_zero()
                check_Js(J)
                expr = reshape_number(var, expr)
                if J.sparsity().get_col():
                    U0[J.sparsity().get_col()] = expr[J.row()]
            else:
                assert not depends_on(var, stage.u)
                J, r = linear_coeffs(var,self.v)
                J = evalf(J)
                r = evalf(r)
                assert r.is_zero()
                check_Js(J)
                expr = reshape_number(var, expr)
                if J.sparsity().get_col():
                    V0[J.sparsity().get_col()] = expr[J.row()]
        self.wrapper.add_declarations(
        f"""
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


        self.wrapper.add_declarations("void preamble(typeUSERPARAM* userparam) {\n")
        for l in self.preamble:
            self.wrapper.add_declarations("  " + l + "\n")
        self.wrapper.add_declarations(f"""
        {self.user}->arg = malloc(sizeof(const casadi_real*)*sz_arg);
        {self.user}->res = malloc(sizeof(casadi_real*)*sz_res);
        {self.user}->iw = sz_iw>0 ? malloc(sizeof(casadi_int)*sz_iw) : 0;
        {self.user}->w = sz_w>0 ? malloc(sizeof(casadi_real)*sz_w) : 0;
        {self.user}->x_opt = malloc(sizeof(casadi_real)*{stage.nx*self.Nhor});
        {self.user}->u_opt = malloc(sizeof(casadi_real)*{stage.nu*self.Nhor});
        {self.user}->v_opt = malloc(sizeof(casadi_real)*{self.v.numel()});
        {self.user}->x_current = malloc(sizeof(casadi_real)*{max(stage.nx, 1)});
        {self.user}->p = malloc(sizeof(casadi_real)*{max(stage.np, 1)});
        {self.user}->umin = malloc(sizeof(casadi_real)*{max(stage.nu, 1)});
        {self.user}->umax = malloc(sizeof(casadi_real)*{max(stage.nu, 1)});
        {self.user}->u0 = malloc(sizeof(casadi_real)*{max(stage.nu, 1)});
        {self.user}->v0 = malloc(sizeof(casadi_real)*{max(self.v.numel(), 1)});
        """)
        self.wrapper.add_declarations("}\n")

        self.wrapper.add_declarations("void postamble(typeUSERPARAM* userparam) {\n")
        for l in self.postamble:
            self.wrapper.add_declarations("  " + l + "\n")
        self.wrapper.add_declarations(f"""
        free({self.user}->arg);
        free({self.user}->res);
        free({self.user}->iw);
        free({self.user}->w);
        free({self.user}->x_opt);
        free({self.user}->u_opt);
        free({self.user}->v_opt);
        free({self.user}->x_current);
        free({self.user}->p);
        free({self.user}->umin);
        free({self.user}->umax);
        free({self.user}->u0);
        free({self.user}->v0);
        }}
        """)

        nc = vertcat(eq,ineq,eq_term,ineq_term).numel()
        vector_options = {"ConstraintsAbsTol": nc}
        for k, L in sorted(vector_options.items()):
            if k in self.grampc_options:
                if isinstance(self.grampc_options[k],float):
                    self.grampc_options[k] = [self.grampc_options[k]]*L
                self.wrapper.add_init_mem(f"""double {k}[{L}] = {{{strlist(self.grampc_options[k])}}};\n""")


        for k,v in stage._param_vals.items():
            self.set_value(stage, self, k, v)

        res = self.pmap(p=self.P0)

        p = self.P0.nonzeros()
        x_current = res["x_current"].nonzeros()
        umax = res["umax"].nonzeros()
        umin = res["umin"].nonzeros()
        u0 = U0.nonzeros()
        v0 = V0.nonzeros()
        Tmin = float(res["Tmin"])
        Tmax = float(res["Tmax"])
        self.wrapper.add_init_mem(f"""
            int i;
            typeUSERPARAM* userparam = malloc(sizeof(cs_struct));
            double x0[{stage.nx}] = {{{strlist(x_current)}}};
            double umax[{stage.nu}] = {{{strlist(umax)}}};
            double umin[{stage.nu}] = {{{strlist(umin)}}};
            double u0[{stage.nu}] = {{{strlist(u0)}}};
            double p[{stage.np}] = {{{strlist(p)}}};
            double v0[{self.v.numel()}] = {{{strlist(v0)}}};
            preamble(userparam);

            for (i=0;i<{stage.np};++i) {self.user}->p[i] = p[i];

            /********* grampc init *********/
            grampc_init(&grampc, userparam);

            grampc_setparam_real_vector(grampc, "x0", x0);
            grampc_setparam_real_vector(grampc, "umax", umax);
            grampc_setparam_real_vector(grampc, "umin", umin);
            grampc_setparam_real_vector(grampc, "u0", u0);
            grampc_setparam_real_vector(grampc, "p0", v0);

            //grampc_setparam_real_vector(grampc, "xdes", 0);
            //grampc_setparam_real_vector(grampc, "udes", 0);

            grampc_setparam_real(grampc, "Thor", {self.control_grid_init[-1]-self.control_grid_init[0]});

            grampc_setparam_real(grampc, "dt", {self.control_grid_init[1]-self.control_grid_init[0]});
            grampc_setparam_real(grampc, "t0", {self.control_grid_init[0]});

            grampc_setopt_int(grampc, "Nhor", {self.Nhor});

            /********* Option definition *********/

            grampc_setopt_string(grampc, "OptimTime", "{"on" if self.free_time else "off"}");
            grampc_setopt_string(grampc, "OptimParam", "{"on" if self.v.numel()>0 else "off"}");
            """)

        if self.free_time:
            self.wrapper.add_init_mem(f"""
                grampc_setparam_real(grampc, "Tmin", {Tmin});
                grampc_setparam_real(grampc, "Tmax", {Tmax});
            """)

        for k,v in sorted(self.grampc_options.items()):
            if k in vector_options.keys():
                self.wrapper.add_init_mem(f"grampc_setopt_real_vector(grampc, \"{k}\", {k});\n")
            if isinstance(v, int):
                self.wrapper.add_init_mem(f"grampc_setopt_int(grampc, \"{k}\", {v});\n")
            elif isinstance(v, float):
                self.wrapper.add_init_mem(f"grampc_setopt_real(grampc, \"{k}\", {v});\n")
            elif isinstance(v, str):
                self.wrapper.add_init_mem(f"grampc_setopt_string(grampc, \"{k}\", \"{v}\");\n")


        self.wrapper.add_init_mem(f"""
            /********* estimate and set PenaltyMin *********/
            grampc_printopt(grampc);
            grampc_printparam(grampc);
            grampc_estim_penmin(grampc, 1);
        """)
        if self.verbose:
            self.wrapper.add_init_mem(f"""
                grampc_printopt(grampc);
                grampc_printparam(grampc);
            """)

        grampc = self.wrapper.codegen_mem(self.wrapper_codegen)
        stats = self.user_grampc+"->stats"
        self.wrapper.add_declarations(f"""
            int grampc_driver_get_stats2(void) {{

                }}
            const compat_solver_stats* grampc_driver_get_stats(void) {{
                typeGRAMPC *grampc = {grampc};
                static compat_solver_stats ret;
                ret.n_sqp_iter = {stats}.n_outer_iter;
                ret.n_qp_iter = {stats}.n_inner_iter;
                ret.sqp_stop_crit = {stats}.conv_grad + 2*{stats}.conv_con;
                return &ret;
            }}
        
        """)


        self.wrapper.set_sp_in([Sparsity.dense(stage.np)])
        self.wrapper.set_name_in(["p"])

        """
        self.set_matrices()
        self._solve(self.grampc)
        x_opt = np.zeros((stage.nx, self.N+1),dtype=np.float64,order='F')
        self._read_x_opt(self.grampc, x_opt.ctypes.data_as(POINTER(c_double)))
        u_opt = np.zeros((stage.nu, self.N+1),dtype=np.float64,order='F')
        self._read_u_opt(self.grampc, u_opt.ctypes.data_as(POINTER(c_double)))
        print("u_opt",u_opt)
        u_opt = u_opt[:,:-1]
        v_opt = np.zeros((self.v.numel()),dtype=np.float64,order='F')
        self._read_v_opt(self.grampc, v_opt.ctypes.data_as(POINTER(c_double)))
        T_opt = self._read_T_opt(self.grampc)
        obj = np.zeros((1),dtype=np.float64)
        Nouter = np.zeros((1),dtype=np.int64)
        Ninner = np.zeros((1),dtype=np.int64)
        conv_grad = np.zeros((1),dtype=np.int64)
        conv_con = np.zeros((1),dtype=np.int64)
        """


        self.wrapper.set_sp_out([Sparsity.dense(stage.nx, self.N+1), Sparsity.dense(stage.nu, self.N), Sparsity.dense(self.v.numel()), Sparsity.dense(1,1)])
        self.wrapper.set_name_out(["x_opt","u_opt","v_opt","T_opt"])

 
        self.wrapper.add_body(f"""
            int mm;
            const casadi_real ** pmap_arg = arg+{self.wrapper.n_in()};
            casadi_real ** pmap_res = res+{self.wrapper.n_out()};
            pmap_arg[0] = arg[0];

            for (int i=0;i<{stage.np};++i) {self.user_grampc}->p[i] = pmap_arg[0][i];
            """)
        for i in range(self.pmap.n_out()):
            n = self.pmap.name_out(i)
            self.wrapper.add_body(f"""pmap_res[{i}] = {"&" if n.startswith("T") else ""}{self.user_grampc}->{n};\n""")

        def lookup_out(name):
            return "res[" + str(["x_opt","u_opt","v_opt","T_opt"].index(name))+"]"

        self.wrapper.add_body(f"""
            mm = pmap_checkout();
            pmap(pmap_arg, pmap_res, iw, w, 0);
            pmap_release(mm);

            grampc_setparam_real_vector(grampc, "x0", {self.user_grampc}->x_current);
            grampc_setparam_real_vector(grampc, "umin", {self.user_grampc}->umin);
            grampc_setparam_real_vector(grampc, "umax", {self.user_grampc}->umax);
            grampc_setparam_real(grampc, "Tmin", {self.user_grampc}->Tmin);
            grampc_setparam_real(grampc, "Tmax", {self.user_grampc}->Tmax);

            grampc_run(grampc);
            grampc_printstatus(grampc->sol->status, STATUS_LEVEL_DEBUG);

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
            for (int i=0;i<{self.v.numel()};++i) {{
                {self.user_grampc}->v_opt[i] = grampc->rws->p[i]*grampc->opt->pScale[i]+grampc->opt->pOffset[i];
            }}
            {self.user_grampc}->T_opt = grampc->rws->T;
            printf("J %f %f %f\\n",grampc->sol->J[0],grampc->sol->J[1],{self.user_grampc}->x_opt[0]);

            if ({lookup_out("x_opt")}) for (int i=0;i<{stage.nx*self.Nhor};++i) {lookup_out("x_opt")}[i] = {self.user_grampc}->x_opt[i];
            if ({lookup_out("u_opt")}) for (int i=0;i<{stage.nu*self.N};++i) {lookup_out("u_opt")}[i] = {self.user_grampc}->u_opt[i];
            if ({lookup_out("v_opt")}) for (int i=0;i<{self.v.numel()};++i) {lookup_out("v_opt")}[i] = {self.user_grampc}->v_opt[i];
            if ({lookup_out("T_opt")}) {lookup_out("T_opt")}[0] = {self.user_grampc}->T_opt;

            for ({self.user_grampc}->stats.n_outer_iter=0,{self.user_grampc}->stats.n_inner_iter=0;{self.user_grampc}->stats.n_outer_iter<grampc->opt->MaxMultIter;++{self.user_grampc}->stats.n_outer_iter) {{
                int n = grampc->sol->iter[{self.user_grampc}->stats.n_outer_iter];
                {self.user_grampc}->stats.n_inner_iter += n;
                if (n==0) break;
            }}

            {self.user_grampc}->stats.conv_grad = convergence_test_gradient(grampc->opt->ConvergenceGradientRelTol, grampc);
            {self.user_grampc}->stats.conv_con = convergence_test_constraints(grampc->opt->ConstraintsAbsTol, grampc);

        """)
        
        build_dir_abs = "foobar"
        self.build_dir_abs = build_dir_abs
        prepare_build_dir(build_dir_abs)

        self.codegen.generate(build_dir_abs+os.sep)

        self.wrapper_codegen.add_dependency(self.wrapper)
        self.wrapper_codegen.generate(build_dir_abs+os.sep)

        
        for e in glob.glob("/home/jgillis/programs/grampc-code/include/*.h"):
            self.artifacts.append(SourceArtifact(e))
        for e in glob.glob("/home/jgillis/programs/grampc-code/src/*.c"):
            self.artifacts.append(SourceArtifact(e))

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
            #add_executable({self.grampc_driver}_main {self.grampc_driver}.c {self.codegen_name}.c)
            
            target_link_libraries({self.grampc_driver} grampc)
            #target_link_libraries({self.grampc_driver}_main grampc)

            install(TARGETS {self.grampc_driver} RUNTIME DESTINATION . LIBRARY DESTINATION .)

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
        subprocess.run(["bash","build.sh"],cwd=build_dir_abs)

        if os.name == "nt":
            libname = self.grampc_driver+".dll"
        else:
            libname = "lib"+self.grampc_driver+".so"
        self.solver = external(self.grampc_driver, libname)

    def to_function(self, stage, name, args, results, *margs):
        print("args=",args)

        res = self.solver(p=stage.p)
        print(stage.p)
        print([stage.value(a) for a in args])


        [_,states] = stage.sample(stage.x,grid='control')
        [_,controls] = stage.sample(stage.u,grid='control-')
        variables = stage.value(vvcat(stage.variables['']))

        helper_in = [states,controls,variables, stage.T]
        helper = Function("helper", helper_in, results)

        arg_in = helper(res["x_opt"],res["u_opt"],res["v_opt"],res["T_opt"])

        ret = Function(name, args, arg_in, *margs)
        assert not ret.has_free()
        return ret

    def initial_value(self, stage, expr):
        ret = self.pmap(p=self.P0)
        parameters = []
        for p in stage.parameters['']:
            parameters.append(stage.value(p))
        
        [_,states] = stage.sample(stage.x,grid='control')
        [_,controls] = stage.sample(stage.u,grid='control-')
        variables = stage.value(vvcat(stage.variables['']))

        helper_in = [vvcat(parameters),states,controls,variables, stage.T]
        helper = Function("helper", helper_in, [expr])
        return helper(self.P0, cs.repmat(ret["x_current"], 1, self.N+1), cs.repmat(self.U0, 1, self.N), self.V0, 0).toarray(simplify=True)

    def solve(self, stage,limited=False):
        ret = self.solver(p=self.P0)

        self.last_solution = OcpSolution(SolWrapper(self, vec(ret["x_opt"]), vec(ret["u_opt"]), ret["v_opt"], ret["T_opt"], rT=stage.T), stage)
        """if not conv:
            if Nouter==self.grampc_options["MaxMultIter"]:
                if not limited:
                    raise Exception("MaxMultIter exhausted without meeting convergence criteria")
            else:
                raise Exception("Problem not converged")
                """
        return self.last_solution

    def non_converged_solution(self, stage):
        return self.last_solution

    def solve_limited(self, stage):
        return self.solve(stage,limited=True)

    def eval(self, stage, expr):
        placeholders = stage.placeholders_transcribed
        expr = placeholders(expr,max_phase=1)
        ks = [self.v,stage.T]
        vs = [self.V_gist, self.T_gist]
        ret = substitute([expr],ks,vs)[0]
        return ret
        
    @property
    def gist(self):
        return vertcat(ExternalMethod.gist.fget(self), self.V_gist, self.T_gist)

    def eval_at_control(self, stage, expr, k):
        placeholders = stage.placeholders_transcribed
        expr = placeholders(expr,max_phase=1)
        ks = [stage.x,stage.u,self.v,stage.T]
        vs = [self.X_gist[k], self.U_gist[min(k, self.N-1)], self.V_gist, self.T_gist]
        if not self.t_state:
            ks += [stage.t]
            vs += [self.control_grid[k]]
        ret = substitute([expr],ks,vs)[0]
        return ret


    @property
    def code_generator_class(self):
        return GrampcCodeGenerator


class SolWrapper:
    def __init__(self, method, x, u, v, T, rT=None):
        self.method = method
        self.x = x
        self.u = u
        self.T = T
        self.v = v
        self.rT = rT

    def value(self, expr, *args,**kwargs):
        placeholders = self.method.stage.placeholders_transcribed
        expr = substitute(expr,self.rT, self.T)
        ret = evalf(substitute([placeholders(expr)],[self.method.gist],[vertcat(self.x, self.u, self.v, self.T)])[0])
        return ret.toarray(simplify=True)