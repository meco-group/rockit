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

from casadi import Opti, jacobian, dot, hessian, symvar, evalf, veccat, DM, vertcat, is_equal
import casadi
import numpy as np
from .casadi_helpers import get_meta, merge_meta, single_stacktrace, MX
from .solution import OcpSolution
from .freetime import FreeTime

class DirectMethod:
    """
    Base class for 'direct' solution methods for Optimal Control Problems:
      'first discretize, then optimize'
    """
    def __init__(self):
        self._solver = None
        self._solver_options = None
        self._callback = None
        self.artifacts = []
        self.clean()

    def clean(self):
        self.V = None
        self.P = []

    def jacobian(self, with_label=False):
        J = jacobian(self.opti.g, self.opti.x).sparsity()
        if with_label:
            return J, "Constraint Jacobian: " + J.dim(True)
        else:
            return J

    def hessian(self, with_label=False):
        lag = self.opti.f + dot(self.opti.lam_g, self.opti.g)
        H = hessian(lag, self.opti.x)[0].sparsity()
        if with_label:
            return H, "Lagrange Hessian: " + H.dim(True)
        else:
            return H

    def spy_jacobian(self):
        import matplotlib.pylab as plt
        J, title = self.jacobian(with_label=True)
        plt.spy(np.array(J))
        plt.title(title)

    def spy_hessian(self):
        import matplotlib.pylab as plt
        lag = self.opti.f + dot(self.opti.lam_g, self.opti.g)
        H, title = self.hessian(with_label=True)
        plt.spy(np.array(H))
        plt.title(title)
    
    def inherit(self, template):
        if template and template._solver is not None:
            self._solver = template._solver
        if template and template._solver_options is not None:
            self._solver_options = template._solver_options
        if template and template._callback is not None:
            self._callback = template._callback

    def eval(self, stage, expr):
        return self.eval_top(stage, expr)

    def eval_top(self, stage, expr):
        return substitute(MX(expr),veccat(*(stage.variables[""]+stage.parameters[""])),vertcat(self.V,veccat(*self.P)))

    def add_variables(self, stage, opti):
        V = []
        for v in stage.variables['']:
            V.append(opti.variable(v.shape[0], v.shape[1], scale=stage._scale[v], domain=stage._catalog[v]['domain']))
        self.V = veccat(*V)

    def add_parameters(self, stage, opti):
        self.P = []
        for p in stage.parameters['']:
            self.P.append(opti.parameter(p.shape[0], p.shape[1]))

    def untranscribe(self, stage, phase=1,**kwargs):
        self.clean()

    def untranscribe_placeholders(self, phase, stage):
        pass

    def main_untranscribe(self,stage, phase=1, **kwargs):
        self.opti = None

    def main_transcribe(self, stage, phase=1, **kwargs):
        if phase==0: return
        if phase==1:
            self.opti = OptiWrapper(stage)
            if self._callback:
                self.opti.callback(self._callback)
            if self.solver is not None:
                if self._solver is None:
                    raise Exception("You forgot to declare a solver. Use e.g. ocp.solver('ipopt').")
                self.opti.solver(self._solver, self._solver_options)
        if phase==2:
            self.opti.transcribe_placeholders(phase, kwargs["placeholders"])

    def transcribe(self, stage, phase=1, **kwargs):
        if stage.nx>0 or stage.nu>0:
            raise Exception("You forgot to declare a method. Use e.g. ocp.method(MultipleShooting(N=3)).")
        if phase==0: return
        if phase>1: return
        self.add_variables(stage, self.opti)
        self.add_parameters(stage, self.opti)

        for c, m, _ in stage._constraints["point"]:
            self.opti.subject_to(self.eval_top(stage, c), meta = m)
        self.opti.add_objective(self.eval_top(stage, stage._objective))
        self.set_initial(stage, self.opti, stage._initial)
        self.set_parameter(stage, self.opti)

    def set_initial(self, stage, master, initial):
        opti = master.opti if hasattr(master, 'opti') else master
        opti.cache_advanced()
        for var, expr in initial.items():
            opti_initial = opti.initial()
            target = self.eval_top(stage, var)
            value = DM(opti.debug.value(self.eval_top(stage, expr), opti_initial)) # HOT line
            opti.set_initial(target, value, cache_advanced=True)

    def set_parameter(self, stage, opti):
        for i, p in enumerate(stage.parameters['']):
            opti.set_value(self.P[i], stage._param_value(p))

    def set_value(self, stage, master, parameter, value):
        opti = master.opti if hasattr(master, 'opti') else master
        found = False
        for i, p in enumerate(stage.parameters['']):
            if is_equal(parameter, p):
                found = True
                opti.set_value(self.P[i], value)
        assert found, "You attempted to set the value of a non-parameter."

    def transcribe_placeholders(self, phase, stage, placeholders):
        pass

    def non_converged_solution(self, stage):
        if not hasattr(self, 'opti'):
            raise Exception("You forgot to solve first. To avoid your script halting, use a try-catch block.")
        return OcpSolution(self.opti.non_converged_solution, stage)

    def solve(self, stage):
        return OcpSolution(self.opti.solve(), stage)

    def solve_limited(self, stage):
        return OcpSolution(self.opti.solve_limited(), stage)

    def callback(self, stage, fun):
        self._callback = lambda iter : fun(iter, OcpSolution(self.opti.non_converged_solution, stage))

    @property
    def debug(self):
        self.opti.debug

    def solver(self, solver, solver_options={}):
        self._solver = solver
        self._solver_options = solver_options

    def show_infeasibilities(self, *args):
        self.opti.debug.show_infeasibilities(*args)

    def initial_value(self, stage, expr):
        return self.opti.value(expr, self.opti.initial())

    @property
    def gist(self):
        """Obtain an expression packing all information needed to obtain value/sample

        The composition of this array may vary between rockit versions

        Returns
        -------
        :obj:`~casadi.MX` column vector

        """
        return vertcat(self.opti.x, self.opti.p)

    def to_function(self, stage, name, args, results, *margs):
        return self.opti.to_function(name, [stage.value(a) for a in args], results, *margs)

    def fill_placeholders_integral(self, phase, stage, expr, *args):
        if phase==1:
            I = stage.state(quad=True)
            stage.set_der(I, expr)
            return stage.at_tf(I)

    def fill_placeholders_T(self, phase, stage, expr, *args):
        if phase==1:
            if isinstance(stage._T, FreeTime):
                init = stage._T.T_init
                stage.set_T(stage.variable())
                stage.subject_to(stage._T>=0)
                stage.set_initial(stage._T, init,priority=True)
                return stage._T
            else:
                return stage._T
        return self.eval(stage, expr)

    def fill_placeholders_t0(self, phase, stage, expr, *args):
        if phase==1:
            if isinstance(stage._t0, FreeTime):
                init = stage._t0.T_init
                stage.set_t0(stage.variable())
                stage.set_initial(stage._t0, init,priority=True)
                return stage._t0
            else:
                return stage._t0
        return self.eval(stage, expr)

    def fill_placeholders_t(self, phase, stage, expr, *args):
        return None
    
    def fill_placeholders_DT(self, phase, stage, expr, *args):
        return None

    def fill_placeholders_DT_control(self, phase, stage, expr, *args):
        return None


from casadi import substitute

class OptiWrapper(Opti):
    def __init__(self, ocp):
        self.ocp = ocp
        Opti.__init__(self)
        self.initial_keys = []
        self.initial_values = []
        self.constraints = []
        self.objective = 0

    def subject_to(self, expr=None, scale=1, meta=None):
        meta = merge_meta(meta, get_meta())
        if expr is None:
            self.constraints = []
        else:
            if isinstance(expr,MX) and expr.is_constant():
                if np.all(np.array(evalf(expr)).squeeze()==1):
                    return
                else:
                    raise Exception("You have a constraint that is never statisfied.")
            self.constraints.append((expr, scale, meta))

    def add_objective(self, expr):
        self.objective = self.objective + expr

    def clear_objective(self):
        self.objective = 0

    def callback(self,fun):
        Opti.callback(self, fun)

    def initial(self):
        return [e for e in Opti.initial(self) if e.dep(0).is_symbolic() or e.dep(1).is_symbolic()]+Opti.value_parameters(self)

    @property
    def non_converged_solution(self):
        return OptiSolWrapper(self, self.debug)

    def variable(self,n=1,m=1, scale=1,domain='real'):
        if n==0 or m==0:
            return MX(n, m)
        else:
            v = Opti.variable(self,n, m)
            if hasattr(Opti,'set_domain'):
                Opti.set_domain(self, v, domain)
            else:
                if domain!='real': raise Exception("This version of CasADi Opti stack does not support set_domain.")
            return scale*v

    def cache_advanced(self):
        self._advanced_cache = self.advanced

    def set_initial(self, key, value, cache_advanced=False):
        a = set([hash(e) for e in (self._advanced_cache if cache_advanced else self.advanced).symvar()])
        b = set([hash(e) for e in symvar(key)])
        if len(a | b)==len(a):
            Opti.set_initial(self, key, value) # set_initial logic in direct_collocation needs this
        else:
            self.initial_keys.append(key)
            self.initial_values.append(value)

    def transcribe_placeholders(self,phase,placeholders):
        opti_advanced = self.advanced
        Opti.subject_to(self)
        n_constr = len(self.constraints)
        res = placeholders([c[0] for c in self.constraints] + [self.objective]+self.initial_keys)
        for c, scale, meta in zip(res[:n_constr], [c[1] for c in self.constraints], [c[2] for c in self.constraints]):
            try:
                if MX(c).is_constant() and MX(c).is_one():
                    continue
                if not MX(scale).is_one():
                    mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
                    if mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY, casadi.OPTI_DOUBLE_INEQUALITY]:
                        print(mc.lb,mc.canon,mc.ub)
                        lb = mc.lb/scale
                        canon = mc.canon/scale
                        ub = mc.ub/scale
                        # Check for infinities
                        try:
                            lb_inf = np.all(np.array(evalf(lb)==-np.inf))
                        except:
                            lb_inf = False
                        try:
                            ub_inf = np.all(np.array(evalf(ub)==np.inf))
                        except:
                            ub_inf = False
                        if lb_inf:
                            c = canon <= ub
                        elif ub_inf:
                            c = lb <= canon
                        else:     
                            c = lb <= (canon <= ub)
                    if mc.type in [casadi.OPTI_EQUALITY, casadi.OPTI_GENERIC_EQUALITY]:
                        lb = mc.lb/scale
                        canon = mc.canon/scale
                        c = lb==canon
                Opti.subject_to(self,c)
            except Exception as e:
                print(meta,c)
                raise e
            self.update_user_dict(c, single_stacktrace(meta))
        Opti.minimize(self,res[n_constr])
        for k_orig, k, v in zip(self.initial_keys,res[n_constr+1:],self.initial_values):
            Opti.set_initial(self, k, v)


    def solve(self):
        return OptiSolWrapper(self, Opti.solve(self))

class OptiSolWrapper:
    def __init__(self, opti_wrapper, sol):
        self.opti_wrapper = opti_wrapper
        self.sol = sol

    def value(self, expr, *args,**kwargs):
        placeholders = self.opti_wrapper.ocp.placeholders_transcribed
        return self.sol.value(placeholders(expr), *args, **kwargs)

    def stats(self):
        return self.sol.stats()