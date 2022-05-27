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

from ..multiple_shooting import MultipleShooting
from ..sampling_method import SamplingMethod, UniformGrid
from ..solution import OcpSolution
from ..freetime import FreeTime
from ..casadi_helpers import DM2numpy, reshape_number
from collections import OrderedDict
from casadi import SX, Sparsity, MX, vcat, veccat, symvar, substitute, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag
import casadi
import numpy as np
import scipy

INF = 1e5

def linear_coeffs(expr, *args):
    """ Multi-argument extesion to CasADi linear_coeff"""
    J,c = linear_coeff(expr, vcat(args))
    cs = np.cumsum([0]+[e.numel() for e in args])
    return tuple([J[:,cs[i]:cs[i+1]] for i in range(len(args))])+(c,)

class ExternalMethod:
    def __init__(self,N=20,grid=UniformGrid(),linesearch=True,expand=False,**args):
        self.N = N
        self.args = args
        self.grid = grid
        self.T = None
        self.linesearch = linesearch
        self.expand = expand

    def inherit(self, parent):
        pass

    def fill_placeholders_t0(self, phase, stage, expr, *args):
        if phase==1:
            if isinstance(stage._t0,FreeTime):
                return stage.at_t0(self.t)
            else:
                return stage._t0
        return

    def fill_placeholders_T(self, phase, stage, expr, *args):
        if phase==1:
            if isinstance(stage._T,FreeTime):
                T = stage.state()
                stage.set_der(T, 0)
                self.T = T
                stage.set_initial(T, stage._T.T_init)
                return stage.at_tf(T)
            else:
                self.T = stage._T
                return stage._T
        return

    def fill_placeholders_t(self, phase, stage, expr, *args):
        if phase==1:
            if self.t_state:
                t = stage.state()
                stage.set_der(t, 1)
                self.t = t
                if not isinstance(stage._t0,FreeTime):
                    stage.subject_to(stage.at_t0(self.t)==stage._t0)
                return t
            else:
                self.t = MX(1,1)
                return
        return

    def transcribe_placeholders(self, phase, stage, placeholders):
        """
        Transcription is the process of going from a continuous-time OCP to an NLP
        """
        return stage._transcribe_placeholders(phase, self, placeholders)
 
    def fill_placeholders_integral(self, phase, stage, expr, *args):
        if phase==1:
            I = stage.state()
            stage.set_der(I, expr)
            stage.subject_to(stage.at_t0(I)==0)
            return stage.at_tf(I)

    def fill_placeholders_sum_control(self, phase, stage, expr, *args):
        if phase==1:
            return expr

    def fill_placeholders_at_t0(self, phase, stage, expr, *args):
        if phase==1: return
        ret = {}
        ks = [stage.x,stage.u]
        vs = [self.X_gist[0],self.U_gist[0]]
        if not self.t_state:
            ks += [stage.t]
            vs += [self.control_grid[0]]
        ret["normal"] = substitute([expr],ks,vs)[0]
        ret["expose"] = expr
        return ret

    def fill_placeholders_at_tf(self, phase, stage, expr, *args):
        if phase==1: return
        ret = {}
        ks = [stage.x,stage.u]
        vs = [self.X_gist[-1],self.U_gist[-1]]
        if not self.t_state:
            ks += [stage.t]
            vs += [self.control_grid[-1]]
        ret["normal"] = substitute([expr],ks,vs)[0]
        ret["expose"] = expr
        return ret

    def main_transcribe(self, stage, phase=1, **kwargs):
        pass

    def transcribe(self, stage, phase=1, **kwargs):

        if phase==0:

            def recursive_depends(expr,t):
                expr = MX(expr)
                if depends_on(expr, t):
                    return True
                if expr in stage._placeholders:
                    if recursive_depends(stage._placeholders[expr][1],t):
                        return True
                else:
                    if expr.is_symbolic():
                        return depends_on(expr, t)
                    for s in symvar(expr):
                        if recursive_depends(s, t):
                            return True
                return False 
                  
            # Do we need to introduce a helper state for t?
            f = stage._ode()
            if f.sparsity_in('t').nnz()>0:
                self.t_state = True
                return

            # Occurs in lagrange?
            for e in symvar(stage._objective):
                if e.name()=='sum_control' and recursive_depends(e,stage.t):
                    self.t_state = True
                    return

            if isinstance(stage._t0, FreeTime):
                self.t_state = True
                return

            if isinstance(stage._T, FreeTime) or isinstance(stage._t0, FreeTime):
                for c,_,_ in stage._constraints["control"]+stage._constraints["integrator"]:
                    if recursive_depends(c,stage.t):
                        self.t_state = True
                        return
            
            self.t_state = False
            return

        if phase==1:
            self.transcribe_phase1(stage, **kwargs)
        else:
            self.transcribe_phase2(stage, **kwargs)

    def set_value(self, stage, master, parameter, value):
            J = jacobian(parameter, stage.p)
            v = reshape_number(parameter, value)
            self.P0[J.sparsity().get_col()] = v[J.row()]

    @property
    def gist(self):
        """Obtain an expression packing all information needed to obtain value/sample

        The composition of this array may vary between rockit versions

        Returns
        -------
        :obj:`~casadi.MX` column vector

        """
        return vertcat(vcat(self.X_gist), vcat(self.U_gist))

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