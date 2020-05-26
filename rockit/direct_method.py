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

from casadi import Opti, jacobian, dot, hessian, symvar, evalf, veccat, DM
import numpy as np
from .casadi_helpers import get_meta, merge_meta, single_stacktrace, MX

class DirectMethod:
    """
    Base class for 'direct' solution methods for Optimal Control Problems:
      'first discretize, then optimize'
    """
    def __init__(self):
        pass

    def jacobian(self, opti, with_label=False):
        J = jacobian(opti.g, opti.x).sparsity()
        if with_label:
            return J, "Constraint Jacobian: " + J.dim(True)
        else:
            return J

    def hessian(self, opti, with_label=False):
        lag = opti.f + dot(opti.lam_g, opti.g)
        H = hessian(lag, opti.x)[0].sparsity()
        if with_label:
            return H, "Lagrange Hessian: " + H.dim(True)
        else:
            return H

    def spy_jacobian(self, opti):
        import matplotlib.pylab as plt
        J, title = self.jacobian(opti, with_label=True)
        plt.spy(np.array(J),vmin=0,vmax=1)
        plt.title(title)

    def spy_hessian(self, opti):
        import matplotlib.pylab as plt
        lag = opti.f + dot(opti.lam_g, opti.g)
        H, title = self.hessian(opti, with_label=True)
        plt.spy(np.array(H),vmin=0,vmax=1)
        plt.title(title)
    
    def register(self, stage):
        pass

    def eval(self, stage, expr):
        return self.eval_top(stage, expr)

    def eval_top(self, stage, expr):
        return substitute(expr,veccat(*stage.variables[""]),self.V)

    def add_variables(self, stage, opti):
        V = []
        for v in stage.variables['']:
            V.append(opti.variable(v.shape[0], v.shape[1]))
        self.V = veccat(*V)

    def transcribe(self, stage, opti):
        self.add_variables(stage, opti)

        for c, m, _ in stage._constraints["point"]:
            opti.subject_to(self.eval_top(stage, c), meta = m)
        opti.add_objective(stage._objective)
        self.set_initial(stage, opti, stage._initial)

    def set_initial(self, stage, opti, initial):
        opti.cache_advanced()
        for var, expr in initial.items():
            opti_initial = opti.initial()
            target = self.eval_top(stage, var)
            value = DM(opti.debug.value(self.eval_top(stage, expr), opti_initial)) # HOT line
            opti.set_initial(target, value, cache_advanced=True)

    def transcribe_placeholders(self, stage, placeholders):
        pass

from casadi import substitute

class OptiWrapper(Opti):
    def __init__(self, ocp):
        self.ocp = ocp
        Opti.__init__(self)
        self.initial_keys = []
        self.initial_values = []

    def subject_to(self, expr=None, meta=None):
        meta = merge_meta(meta, get_meta())
        if expr is None:
            self.constraints = []
        else:
            if isinstance(expr,MX) and expr.is_constant():
                if np.all(np.array(evalf(expr)).squeeze()==1):
                    return
                else:
                    raise Exception("You have a constraint that is never statisfied.")
            self.constraints.append((expr, meta))

    def add_objective(self, expr):
        self.objective = self.objective + expr

    def clear_objective(self):
        self.objective = 0

    def callback(self,fun):
        Opti.callback(self, fun)

    def initial(self):
        return [e for e in Opti.initial(self) if e.dep(0).is_symbolic() or e.dep(1).is_symbolic()]

    @property
    def non_converged_solution(self):
        return OptiSolWrapper(self, self.debug)

    def variable(self,n=1,m=1):
        if n==0 or m==0:
            return MX(n, m)
        else:
            return Opti.variable(self,n, m)

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

    def transcribe_placeholders(self,placeholders):
        Opti.subject_to(self)
        n_constr = len(self.constraints)
        res = placeholders([c[0] for c in self.constraints] + [self.objective]+self.initial_keys)
        for c, meta in zip(res[:n_constr], [c[1] for c in self.constraints]):
            try:
                if MX(c).is_constant() and MX(c).is_one():
                    continue
                Opti.subject_to(self,c)
            except Exception as e:
                print(meta)
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