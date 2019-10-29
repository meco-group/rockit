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

from casadi import Opti, jacobian, dot, hessian
from .casadi_helpers import get_meta, merge_meta, single_stacktrace, MX

class DirectMethod:
    """
    Base class for 'direct' solution methods for Optimal Control Problems:
      'first discretize, then optimize'
    """
    def __init__(self):
        pass

    def spy_jacobian(self, opti):
        import matplotlib.pylab as plt
        J = jacobian(opti.g, opti.x).sparsity()
        plt.spy(J)
        plt.title("Constraint Jacobian: " + J.dim(True))

    def spy_hessian(self, opti):
        import matplotlib.pylab as plt
        lag = opti.f + dot(opti.lam_g, opti.g)
        H = hessian(lag, opti.x)[0].sparsity()
        plt.spy(H)
        plt.title("Lagrange Hessian: " + H.dim(True))
    
    def register(self, stage):
        pass

    def transcribe(self, stage, opti):
        for c, m, _ in stage._constraints["point"]:
            opti.subject_to(c, meta = m)
        opti.add_objective(stage._objective)

    def transcribe_placeholders(self, stage, placeholders):
        pass

from casadi import substitute

class OptiWrapper(Opti):
    def __init__(self, ocp):
        self.ocp = ocp
        Opti.__init__(self)

    def subject_to(self, expr=None, meta=None):
        meta = merge_meta(meta, get_meta())
        if expr is None:
            self.constraints = []
        else:
            self.constraints.append((expr, meta))

    def add_objective(self, expr):
        self.objective = self.objective + expr

    def clear_objective(self):
        self.objective = 0

    def callback(self,fun):
        super().callback(fun)

    @property
    def non_converged_solution(self):
        return OptiSolWrapper(self, self.debug)

    def variable(self,n=1,m=1):
        if n==0 or m==0:
            return MX(n, m)
        else:
            return Opti.variable(self,n, m)

    def transcribe_placeholders(self,placeholders):
        Opti.subject_to(self)
        res = placeholders([c[0] for c in self.constraints] + [self.objective])
        for c, meta in zip(res[:-1], [c[1] for c in self.constraints]):
            try:
                Opti.subject_to(self,c)
            except Exception as e:
                print(meta)
                raise e
            self.update_user_dict(c, single_stacktrace(meta))
        Opti.minimize(self,res[-1])


    def solve(self):
        return OptiSolWrapper(self, Opti.solve(self))

class OptiSolWrapper:
    def __init__(self, opti_wrapper, sol):
        self.opti_wrapper = opti_wrapper
        self.sol = sol

    def value(self, expr, *args,**kwargs):
        placeholders = self.opti_wrapper.ocp.placeholders_transcribed
        return self.sol.value(placeholders(expr), *args, **kwargs)