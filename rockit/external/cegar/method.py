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
from ...casadi_helpers import prepare_build_dir, ConstraintInspector
from ..method import ExternalMethod, linear_coeffs, check_Js
from ...solution import OcpSolution

import casadi as ca
import numpy as np



def sx_write(expr):
    s = ca.symvar(expr)
    f = ca.Function('f',s,[expr])

    # Input values of the same dimensions as the above
    input_val = [e.name() for e in s]

    # Output values to be calculated of the same dimensions as the above
    output_val = [None]

    # Work vector
    work = [None]*f.sz_w()

    # For debugging
    instr = f.instructions_sx()

    # Loop over the algorithm
    for k in range(f.n_instructions()):

        # Get the atomic operation
        op = f.instruction_id(k)
        o = f.instruction_output(k)
        i = f.instruction_input(k)

        if(op==ca.OP_CONST):
            work[o[0]] = "%.16f" % float(f.instruction_constant(k))
        else:
            if op==ca.OP_INPUT:
                work[o[0]] = input_val[i[0]]
            elif op==ca.OP_OUTPUT:
                output_val[o[0]] = work[i[0]]
            elif op==ca.OP_SQ:
                work[o[0]] = work[i[0]] + "**2"
            else:
                disp_in = [work[a] for a in i]
                debug_str = ca.print_operator(instr[k],disp_in)
                work[o[0]] = debug_str
    return output_val[0]

def get_terms(e):
    def get_terms_internal(e):
        if e.op()==ca.OP_ADD:
            for i in range(e.n_dep()):
                for t in get_terms_internal(e.dep(i)):
                    yield t
        else:
            yield e
    return list(get_terms_internal(e))


def visit(e,parents=None):
    if parents is None:
        parents = []
    yield (e,parents)
    for i in range(e.n_dep()):
        for t in visit(e.dep(i),parents=[e]+parents):
            yield t

class CegarMethod(ExternalMethod):
    def __init__(self,
        **kwargs):

        supported = {"free_T"}
        ExternalMethod.__init__(self, supported=supported, **kwargs)

    def to_sx(self,e,suffix=""):
        ret = []
        if e.is_scalar():
            ret.append(ca.SX.sym(e.name()+suffix))
        else:
            ret.extend(ca.SX.sym(e.name()+("_%d" % i) + suffix) for  i in range(e.numel()))
        return ret

    def to_sxs(self,L,suffix=""):
        ret = []
        for e in L:
            ret+=self.to_sx(e,suffix=suffix)
        return ret

    def get_state_names(self):
        ret = []
        for e in self.stage.states:
            if e.is_scalar():
                ret.append(e.name())
            else:
                ret.extend(e.name()+"_%d" % i for  i in range(e.numel()))
        return ret

    def get_control_names(self):
        ret = []
        for e in self.stage.controls:
            if e.is_scalar():
                ret.append(e.name())
            else:
                ret.extend(e.name()+"_%d" % i for  i in range(e.numel()))
        return ret

    def get_variable_names(self):
        return self.get_state_names()+self.get_control_names()

    def get_constants(self):
        ret = []
        for e in self.stage.parameters['']:
            if e.is_scalar():
                ret.append(e.name()+"=%.16f" % float(self.stage._param_vals[e]))
            else:
                nz = self.stage._param_vals[e].nonzeros()
                ret.extend(e.name()+"_%d=%.16f" % (i,nz[i]) for i in range(e.numel()))
        return ret

    def get_ODE(self):
        res = self.f(x=ca.vcat(self.x_args),u=ca.vcat(self.u_args),p=ca.vcat(self.p_args))["ode"]

        ret = []
        vars = self.get_state_names()
        for i,e in enumerate(vars):
            ret.append(e+"D' == " + sx_write(res[i]))

        return ret
    
    def get_cost(self):
        f = ca.Function('f',[ca.vcat(self.stage.states),ca.vcat(self.stage.controls),ca.vcat(self.stage.parameters[''])],[self.lagrange])
        result  = f(ca.vcat(self.x_args),ca.vcat(self.u_args),ca.vcat(self.p_args))
        return "costD' == " + sx_write(result)

    def fill_placeholders_integral(self, phase, stage, expr, *args):
        if phase==1:
            return expr

    def fill_placeholders_sum_control(self, phase, stage, expr, *args):
        raise Exception("ocp.sum not supported. Use ocp.integral instead.")

    def fill_placeholders_sum_control_plus(self, phase, stage, expr, *args):
        raise Exception("ocp.sum not supported. Use ocp.integral instead.")

    def transcribe_phase1(self, stage, **kwargs):
        self.stage = stage

        self.v = ca.vvcat(stage.variables[''])
        self.X_gist = [ca.MX.sym("Xg", stage.nx) for k in range(self.N+1)]
        self.U_gist = [ca.MX.sym("Ug", stage.nu) for k in range(self.N)]
        self.V_gist = ca.MX.sym("Vg", *self.v.shape)
        self.T_gist = ca.MX.sym("Tg")

        self.constraint_inspector = ConstraintInspector(self, stage)


        #self.time_grid = self.grid(stage._t0, stage._T, self.N)
        self.normalized_time_grid = self.grid(0.0, 1.0, self.N)
        self.time_grid = self.normalized_time_grid
        if not isinstance(stage._T, FreeTime): self.time_grid*= stage._T
        if not isinstance(stage._t0, FreeTime): self.time_grid+= stage._t0
        self.control_grid = ca.MX(stage.t0 + self.normalized_time_grid*stage.T).T

        self.lagrange = ca.MX(0)
        self.mayer = ca.MX(0)
        var_mayer = []
        obj = ca.MX(stage._objective)
        terms = get_terms(obj)
        for term in terms:
            n = [e.name() for e in ca.symvar(term)]
            sumi = np.sum([e=="r_integral" for e in n])
            summ = np.sum([e=="r_at_tf" for e in n])
            if sumi>1:
                raise Exception("Objective cannot be parsed: operation combining two integrals not supported")
            if sumi==1:
                n_hits = 0
                for e,parents in visit(term):
                    if e.is_symbolic() and e.name()=="r_integral":
                        n_hits+=1
                        for pi in range(len(parents)):
                            p = parents[pi]
                            pchild = parents[pi-1] if pi>0 else e
                            correct = False
                            # Distributive operations
                            if p.op()==ca.OP_MUL:
                                correct = True
                            if p.op()==ca.OP_DIV:
                                # Only allow integral in LHS
                                correct = hash(p.dep(0))==hash(pchild)
                            assert correct, "Objective cannot be parsed: integrals can only be multiplied or divided."
                    if n_hits>1:
                        raise Exception("Objective cannot be parsed")
                if summ!=0:
                    raise Exception("Objective cannot be parsed: operation combining integral and at_tf part not supported")
                self.lagrange += term
                continue
            self.mayer += term

    def transcribe_phase2(self, stage, **kwargs):

        self.constraint_inspector.finalize()
        
        placeholders = kwargs["placeholders"]



        # Total Lagrange integrand
        self.lagrange = placeholders(self.lagrange,preference=['expose'])
        # Total Mayer term
        self.mayer = placeholders(self.mayer,preference=['expose'])

        self.f = self.stage._ode()
        self.x_args = self.to_sxs(self.stage.states,suffix="D")
        self.u_args = self.to_sxs(self.stage.controls,suffix="D")
        self.p_args = self.to_sxs(self.stage.parameters[''],suffix="D")




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
        self.solver.generate_in(self.build_dir_abs+os.sep+"debug_in.txt",[self.P0])
        ret = self.solver(p=self.P0)


        self.stats = stats = self._grampc_driver_get_stats_internal().contents
        self.last_solution = OcpSolution(SolWrapper(self, vec(ret["x_opt"]), vec(ret["u_opt"]), ret["v_opt"], ret["T_opt"], rT=stage.T), stage)

        conv = stats.conv_grad and stats.conv_con

        if not conv:
            if stats.n_outer_iter==self.grampc_options["MaxMultIter"]:
                if not limited:
                    raise Exception("MaxMultIter exhausted without meeting convergence criteria")
            else:
                raise Exception("Problem not converged")

        return self.last_solution

    def get_stats(self):
        stats = self.stats
        return dict((k,getattr(stats,k)) for k,_ in stats_fields)

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

    def stats(self):
        return self.method.get_stats()
