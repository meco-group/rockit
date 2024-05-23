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
from ..casadi_helpers import DM2numpy, reshape_number, linear_coeffs
from collections import OrderedDict
from casadi import SX, Sparsity, MX, vcat, veccat, symvar, substitute, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag
import casadi
import numpy as np
import scipy
import casadi as ca

INF = 1e5

def legit_J(J):
    """
    Checks if J, a pre-multiplier for states and control, is of legitimate structure
    J must a slice of a permuted unit matrix.
    """
    try:
        J = evalf(J)
    except:
        return False
    if not np.all(np.array(J.nonzeros())==1): # All nonzeros must be one
        return False
    # Each row must contain exactly 1 nonzero
    if not np.all(np.array(sum2(J))==1):
        return False
    # Each column must contain at most 1 nonzero
    if not np.all(np.array(sum1(J))<=1):
        return False
    return True


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

def fill_in(var, expr,value, array, k):
    value = evalf(value)
    J, r = linear_coeffs(var,expr)
    J = evalf(J)
    r = evalf(r)
    assert r.is_zero()
    check_Js(J)
    expr = reshape_number(var, expr)
    if J.sparsity().get_col():
        array[J.sparsity().get_col(), k] = value[J.row()]
def fill_in_array(var, expr, value, array, numeric = True):
    if numeric:
        value = evalf(value)
    J, r = linear_coeffs(var[:,0],expr)
    J = evalf(J)
    r = evalf(r)
    assert r.is_zero()
    check_Js(J)
    expr = reshape_number(var, expr)
    if J.sparsity().get_col():
        array[J.sparsity().get_col(), :] = value[J.row(), :]

class ExternalMethod:
    def __init__(self,supported=None,N=20,grid=UniformGrid(),linesearch=True,expand=False,**args):
        self.N = N
        self.args = args
        self.grid = grid
        self.T = None
        self.linesearch = linesearch
        self.expand = expand
        self.supported = {} if supported is None else supported
        self.free_time = False

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
                self.free_time = True
                if "free_T" in self.supported:
                    # Keep placeholder symbol
                    return
                else:
                    T = stage.register_state(MX.sym('T'))
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
                stage.set_next(t, t+ stage.DT) if stage._state_next else stage.set_der(t, 1)
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

    def fill_placeholders_DT(self, phase, stage, expr, *args):
        return None

    def fill_placeholders_DT_control(self, phase, stage, expr, *args):
        return None

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
            f = stage._hybrid()[0]
            if not stage._state_next:
                if f.sparsity_in('t').nnz()>0:
                    self.t_state = True
                    return

            # Occurs in lagrange?
            obj = MX(stage._objective)
            for e in symvar(obj):
                if e.name()=='r_sum_control' and recursive_depends(e,stage.t):
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

            self.p_global = stage.parameters['']
            self.p_local = stage.parameters['control']+stage.parameters['control+']

            self.p_global_cat = ca.vvcat(self.p_global)
            self.p_local_cat = ca.vvcat(self.p_local)


            self.p_global_value = DM.zeros(ca.vvcat(self.p_global).sparsity())
            self.p_local_value = DM.zeros(ca.vvcat(self.p_local).numel(), self.N+1)

            self.transcribe_phase1(stage, **kwargs)
        else:
            self.transcribe_phase2(stage, **kwargs)

    def set_value(self, stage, master, parameter, value):
            value = evalf(value)
            if parameter in self.p_global:
                p = ca.vec(parameter)
                fill_in(p, self.p_global_cat, value, self.p_global_value,0)
            elif parameter in self.p_local:
                p = ca.vec(parameter)
                # for k in list(range(self.N)):
                if p.numel()==value.numel():
                    if(value.shape[1] == p.shape[0]):
                        value = value.T
                    value = ca.repmat(value,  1, self.N+1)
                if p.numel()*(self.N)==value.numel() or p.numel()*(self.N+1)== value.numel():
                    if(value.shape[1] == p.shape[0]):
                        value = value.T
                assert(value.shape[0] == p.shape[0])
                if p.numel()*(self.N)== value.numel():
                    value = ca.horzcat(value, value[:,-1])
                fill_in_array(p, self.p_local_cat, value, self.p_local_value)
            else:
                raise Exception("Parameter not found")

    @property
    def gist(self):
        """Obtain an expression packing all information needed to obtain value/sample

        The composition of this array may vary between rockit versions

        Returns
        -------
        :obj:`~casadi.MX` column vector

        """
        ret = []
        for e,_,_ in self.gist_parts:
            if isinstance(e, list):
                ret.append(vcat(e))
            else:
                ret.append(e)
        return vcat(ret)
    

    def eval(self, stage, expr):
        placeholders = stage.placeholders_transcribed
        expr = placeholders(expr,max_phase=1)
        ks = []
        vs = []
        for g,s,mode in self.gist_parts:
            if mode=="global":
                ks.append(s)
                vs.append(g)
        ret = substitute([expr],ks,vs)[0]
        return ret

    def eval_at_control(self, stage, expr, k):
        placeholders = stage.placeholders_transcribed
        expr = placeholders(expr,max_phase=1)
        ks = []
        vs = []
        for g,s,mode in self.gist_parts:
           ks.append(s)
           if isinstance(g, list):
               g = g[min(k,len(g)-1)]
           vs.append(g)
        if not self.t_state:
            ks += [stage.t]
            vs += [self.control_grid[k]]
        ret = substitute([expr],ks,vs)[0]
        return ret


class SolWrapper:
    def __init__(self, method, x, u):
        self.method = method
        self.x = x
        self.u = u

    def value(self, expr, *args,**kwargs):
        placeholders = self.method.stage.placeholders_transcribed
        ret = evalf(substitute([placeholders(expr)],[self.method.gist],[vertcat(self.x, self.u)])[0])
        return ret.toarray(simplify=True)
