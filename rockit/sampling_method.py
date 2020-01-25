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

from __future__ import division

from casadi import integrator, Function, MX, hcat, vertcat, vcat, linspace, veccat, DM, repmat, horzsplit, mtimes, symvar, horzcat
from .direct_method import DirectMethod
from .splines import BSplineBasis, BSpline
from .casadi_helpers import reinterpret_expr
from numpy import nan, inf
import numpy as np


# Agnostic about free or fixed start or end point: 

# Agnostic about time?
class Grid:
    def __init__(self, min=0, max=inf):
        self.min = min
        self.max = max
        self.localize_T = False
        self.localize_t0 = False

    def __call__(self, t0, T, N):
        return linspace(t0, T, N+1)

def FreeGrid(self):
    """Specify a grid with unprescribed spacing
    
    Parameters
    ----------
    min : float or  :obj:`casadi.MX`
        Minimum size of control interval
        Enforced with constraints
        Default: 0
    max : float or  :obj:`casadi.MX`
        Maximum size of control interval
        Enforced with constraints
        Default: inf
    """
    def __init__(self, *args, **kwargs):
        Grid.__init__(self, *args, **kwargs)
        self.localize_T = True

    def constrain_T(self,opti,T,Tnext,N):
        return

class UniformGrid(Grid):
    """Specify a grid with uniform spacing
    
    Parameters
    ----------
    min : float or  :obj:`casadi.MX`
        Minimum size of control interval
        Enforced with constraints
        Default: 0
    max : float or  :obj:`casadi.MX`
        Maximum size of control interval
        Enforced with constraints
        Default: inf
    """
    def __init__(self, *args, **kwargs):
        Grid.__init__(self, *args, **kwargs)

    def constrain_T(self,opti,T,Tnext,N):
        opti.subject_to(T==Tnext)

class GeometricGrid(Grid):
    """Specify a geometrically growing grid
    
    Each control interval is a constant factor larger than its predecessor

    Parameters
    ----------
    growth_factor : float>1
        See `local` for interpretation
    local : bool, optional
        if False, last interval is growth_factor larger than first
        if True, interval k+1 is growth_factor larger than interval k
        Default: False
    min : float or  :obj:`casadi.MX`
        Minimum size of control interval
        Enforced with constraints
        Default: 0
    max : float or  :obj:`casadi.MX`
        Maximum size of control interval
        Enforced with constraints
        Default: inf

    Examples
    --------

    >>> MultipleShooting(N=3, time_grid=GeometricGrid(2)) # grid = [0, 1, 3, 7]*T/7
    >>> MultipleShooting(N=3, time_grid=GeometricGrid(2,local=True)) # grid = [0, 1, 5, 21]*T/21
    """
    def __init__(self, growth_factor, local=False, *args, **kwargs):
        assert growth_factor>=1
        self._growth_factor = growth_factor
        self.local = local
        Grid.__init__(self, *args, **kwargs)

    def growth_factor(self, N):
        if not self.local:
            return self._growth_factor**(1.0/(N-1))
        return self._growth_factor

    def constrain_T(self,opti,T,Tnext,N):
        opti.subject_to(T*self.growth_factor(N)==Tnext)

    def __call__(self, t0, T, N):
        g = self.growth_factor(N)
        base = 1.0
        vec = [0]
        for i in range(N):
            vec.append(vec[-1]+base)
            base *= g
        return t0 + hcat(vec)/vec[-1]*T

#class Density(TimeGrid):
#    def __init__(self, density, **kwargs): # [0,1]->R^+  integral = 1
#        TimeGrid.__init__(self, **kwargs)
#        self.density = density

#class Fraction(TimeGrid):
#    def __init__(self, grid):
#        self.grid = (grid-grid[0])/grid[-1]

class SamplingMethod(DirectMethod):
    def __init__(self, N=50, M=1, intg='rk', intg_options=None, time_grid=UniformGrid(), localize_t0=False, localize_T=False, **kwargs):
        """
        Parameters
        ----------
        grid : `str` or tuple of :obj:`casadi.MX`
            List of arbitrary expression containing states, controls, ...
        name : `str`
            Name for CasADi Function
        options : dict, optional
            Options for CasADi Function
        """
        DirectMethod.__init__(self, **kwargs)
        self.N = N
        self.M = M
        self.intg = intg
        self.intg_options = {} if intg_options is None else intg_options
        self.time_grid = time_grid
        self.localize_t0 = localize_t0
        self.localize_T = localize_T

        self.X = []  # List that will hold N+1 decision variables for state vector
        self.U = []  # List that will hold N decision variables for control vector
        self.Z = []  # Algebraic vars
        self.T = None
        self.t0 = None
        self.P = []
        self.V = None
        self.V_control = []
        self.P_control = []

        self.poly_coeff = []  # Optional list to save the coefficients for a polynomial
        self.poly_coeff_z = []  # Optional list to save the coefficients for a polynomial
        self.xk = []  # List for intermediate integrator states
        self.zk = []
        self.xr = []
        self.zr = []
        self.tr = []
        self.q = 0

    def discrete_system(self, stage):
        f = stage._ode()

        # Coefficient matrix from RK4 to reconstruct 4th order polynomial (k1,k2,k3,k4)
        # nstates x (4 * M)
        poly_coeffs = []

        t0 = MX.sym('t0')
        T = MX.sym('T')
        DT = T / self.M

        # Size of integrator interval
        X0 = f.mx_in("x")            # Initial state
        U = f.mx_in("u")             # Control
        P = f.mx_in("p")
        Z = f.mx_in("z")

        X = [X0]
        if hasattr(self, 'intg_' + self.intg):
            intg = getattr(self, "intg_" + self.intg)(f, X0, U, P, Z)
        else:
            intg = self.intg_builtin(f, X0, U, P, Z)
        assert not intg.has_free()

        # Compute local start time
        t0_local = t0
        quad = 0
        for j in range(self.M):
            intg_res = intg(x0=X[-1], u=U, t0=t0_local, DT=DT, p=P)
            X.append(intg_res["xf"])
            quad = quad + intg_res["qf"]
            poly_coeffs.append(intg_res["poly_coeff"])
            t0_local += DT

        ret = Function('F', [X0, U, T, t0, P], [X[-1], hcat(X), hcat(poly_coeffs), quad],
                       ['x0', 'u', 'T', 't0', 'p'], ['xf', 'Xi', 'poly_coeff', 'qf'])
        assert not ret.has_free()
        return ret

    def intg_rk(self, f, X, U, P, Z):
        assert Z.is_empty()
        DT = MX.sym("DT")
        t0 = MX.sym("t0")
        # A single Runge-Kutta 4 step
        k1 = f(x=X, u=U, p=P, t=t0, z=Z)
        k2 = f(x=X + DT / 2 * k1["ode"], u=U, p=P, t=t0+DT/2, z=Z)
        k3 = f(x=X + DT / 2 * k2["ode"], u=U, p=P, t=t0+DT/2)
        k4 = f(x=X + DT * k3["ode"], u=U, p=P, t=t0+DT)

        f0 = k1["ode"]
        f1 = 2/DT*(k2["ode"]-k1["ode"])/2
        f2 = 4/DT**2*(k3["ode"]-k2["ode"])/6
        f3 = 4*(k4["ode"]-2*k3["ode"]+k1["ode"])/DT**3/24
        poly_coeff = hcat([X, f0, f1, f2, f3])
        return Function('F', [X, U, t0, DT, P], [X + DT / 6 * (k1["ode"] + 2 * k2["ode"] + 2 * k3["ode"] + k4["ode"]), poly_coeff, DT / 6 * (k1["quad"] + 2 * k2["quad"] + 2 * k3["quad"] + k4["quad"])], ['x0', 'u', 't0', 'DT', 'p'], ['xf', 'poly_coeff', 'qf'])

    def intg_builtin(self, f, X, U, P, Z):
        # A single CVODES step
        DT = MX.sym("DT")
        t = MX.sym("t")
        t0 = MX.sym("t0")
        res = f(x=X, u=U, p=P, t=t0+t*DT, z=Z)
        data = {'x': X, 'p': vertcat(U, DT, P, t0), 'z': Z, 't': t, 'ode': DT * res["ode"], 'quad': DT * res["quad"], 'alg': res["alg"]}
        options = dict(self.intg_options)
        if self.intg in ["collocation"]:
            options["number_of_finite_elements"] = 1
        I = integrator('intg_cvodes', self.intg, data, options)
        res = I.call({'x0': X, 'p': vertcat(U, DT, P, t0)})
        return Function('F', [X, U, t0, DT, P], [res["xf"], MX(), res["qf"]], ['x0', 'u', 't0', 'DT', 'p'], ['xf', 'poly_coeff','qf'])

    def transcribe_placeholders(self, stage, placeholders):
        """
        Transcription is the process of going from a continuous-time OCP to an NLP
        """
        return stage._transcribe_placeholders(self, placeholders)

    def transcribe(self, stage, opti):
        """
        Transcription is the process of going from a continuous-time OCP to an NLP
        """

        self.add_variables(stage, opti)
        self.add_parameter(stage, opti)

        # Create time grid (might be symbolic)
        self.T = self.eval(stage, stage._T)
        self.t0 = self.eval(stage, stage._t0)

        self.set_initial(stage, opti, stage._initial)
        T_init = opti.debug.value(self.T, opti.initial())
        t0_init = opti.debug.value(self.t0, opti.initial())

        # How to get initial value -> ask opti?

        control_grid_init = self.time_grid(t0_init, t0_init + T_init, self.N)
        if self.localize_t0 or self.time_grid.localize_t0:
            for k in range(1, self.N):
                stage.set_initial(self.t0_local[k], control_grid_init[k])
            stage.set_initial(self.t0_local[self.N], control_grid_init[self.N])
        if self.localize_T or self.time_grid.localize_T:
            for k in range(1, self.N):
                stage.set_initial(self.T_local[k], control_grid_init[k+1]-control_grid_init[k])

        self.integrator_grid = []
        for k in range(self.N):
            t_local = linspace(self.control_grid[k], self.control_grid[k+1], self.M+1)
            self.integrator_grid.append(t_local[:-1] if k<self.N-1 else t_local)
        self.add_constraints_before(stage, opti)
        self.add_constraints(stage, opti)
        self.add_constraints_after(stage, opti)
        self.add_objective(stage, opti)
        self.set_initial(stage, opti, stage._initial)
        self.set_parameter(stage, opti)


    def add_constraints_before(self, stage, opti):
        for c, meta, _ in stage._constraints["point"]:
            e = self.eval(stage, c)
            if 'r_at_tf' not in [a.name() for a in symvar(e)]:
                opti.subject_to(e, meta=meta)

    def add_constraints_after(self, stage, opti):
        for c, meta, _ in stage._constraints["point"]:
            e = self.eval(stage, c)
            if 'r_at_tf' in [a.name() for a in symvar(e)]:
                opti.subject_to(e, meta=meta)

    def add_inf_constraints(self, stage, opti, c, k, l, meta):
        coeff = stage._method.poly_coeff[k * self.M + l]
        degree = coeff.shape[1]-1
        basis = BSplineBasis([0]*(degree+1)+[1]*(degree+1),degree)
        tscale = self.T / self.N / self.M
        tpower = vcat([tscale**i for i in range(degree+1)])
        coeff = coeff * repmat(tpower.T,stage.nx,1)
        # TODO: bernstein transformation as function of degree
        Poly_to_Bernstein_matrix_4 = DM([[1,0,0,0,0],[1,1.0/4, 0, 0, 0],[1, 1.0/2, 1.0/6, 0, 0],[1, 3.0/4, 1.0/2, 1.0/4, 0],[1, 1, 1, 1, 1]])
        state_coeff = mtimes(Poly_to_Bernstein_matrix_4,coeff.T)
        
        statesize = [0] + [elem.nnz() for elem in stage.states]
        statessizecum = np.cumsum(statesize)

        subst_from = stage.states
        state_coeff_split = horzsplit(state_coeff,statessizecum)
        subst_to = [BSpline(basis,coeff) for coeff in state_coeff_split]
        c_spline = reinterpret_expr(c, subst_from, subst_to)
        opti.subject_to(self.eval_at_control(stage, c_spline, k), meta=meta)

    def fill_placeholders_integral_control(self, stage, expr, *args):
        r = 0
        for k in range(self.N):
            dt = self.control_grid[k + 1] - self.control_grid[k]
            r = r + self.eval_at_control(stage, expr, k)*dt
        return r

    def fill_placeholders_at_t0(self, stage, expr, *args):
        return self.eval_at_control(stage, expr, 0)

    def fill_placeholders_at_tf(self, stage, expr, *args):
        return self.eval_at_control(stage, expr, -1)

    def fill_placeholders_t0(self, stage, expr, *args):
        return self.t0

    def fill_placeholders_T(self, stage, expr, *args):
        return self.T

    def add_objective(self, stage, opti):
        opti.add_objective(self.eval(stage, stage._objective))

    def add_variables_V(self, stage, opti):
        V = []
        for v in stage.variables['']:
            V.append(opti.variable(v.shape[0], v.shape[1]))
        self.V = veccat(*V)

        # Create time grid (might be symbolic)
        self.T = self.eval(stage, stage._T)
        self.t0 = self.eval(stage, stage._t0)

        if self.localize_t0 or self.time_grid.localize_t0:
            self.t0_local = [None]*(self.N+1)
        if self.localize_T or self.time_grid.localize_T:
            self.T_local = [None]*self.N

    def add_variables_V_control(self, stage, opti, k):
        if k==0:
            self.V_control = [[] for v in stage.variables['control']]

        for i, v in enumerate(stage.variables['control']):
            self.V_control[i].append(opti.variable(v.shape[0], v.shape[1]))
    
        if self.localize_t0 or self.time_grid.localize_t0:
            if k==0:
                self.t0_local[k] = self.t0
            else:
                self.t0_local[k] = opti.variable()
        if self.localize_T or self.time_grid.localize_T:
            if k==0:
                self.T_local[k] = self.T/self.N
            else:
                self.T_local[k] = opti.variable()

    def add_variables_V_control_finalize(self, stage, opti):
        if self.localize_t0 or self.time_grid.localize_t0:
            self.t0_local[self.N] = opti.variable()
            self.control_grid = hcat(self.t0_local)
        elif self.localize_T or self.time_grid.localize_T:
            t0 = self.t0
            cumsum = [t0]
            for e in self.T_local:
                cumsum.append(cumsum[-1]+e)
            self.control_grid = hcat(cumsum)
        else:
            self.control_grid = self.time_grid(MX(self.t0), self.t0 + self.T, self.N)

    def add_coupling_constraints(self, stage, opti, k):
        Tk = self.T/self.N
        if self.localize_T or self.time_grid.localize_T:
            Tk = self.T_local[k]
            if k>=0 and k+1<self.N:
                self.time_grid.constrain_T(opti, self.T_local[k], self.T_local[k+1], self.N)
        if k==0:
            opti.subject_to(self.time_grid.min <= (Tk <= self.time_grid.max))
        if (self.localize_t0 or self.time_grid.localize_t0) and k>=0:
            opti.subject_to(self.t0_local[k]+Tk==self.t0_local[k+1])

    def get_p_control_at(self, stage, k=-1):
        return veccat(*[p[:,k] for p in self.P_control])

    def get_v_control_at(self, stage, k=-1):
        return veccat(*[v[k] for v in self.V_control])

    def eval(self, stage, expr):
        return stage._expr_apply(expr, p=veccat(*self.P), v=self.V, t0=self.t0, T=self.T)

    def eval_at_control(self, stage, expr, k):
        return stage._expr_apply(expr, t0=self.t0, T=self.T, x=self.X[k], z=self.Z[k] if self.Z else nan, xq=self.q if k==-1 else nan, u=self.U[k], p_control=self.get_p_control_at(stage, k), v=self.V, p=veccat(*self.P), v_control=self.get_v_control_at(stage, k), t=self.control_grid[k])

    def eval_at_integrator(self, stage, expr, k, i):
        return stage._expr_apply(expr, t0=self.t0, T=self.T, x=self.xk[k*self.M + i], z=self.zk[k*self.M + i] if self.zk else nan, u=self.U[k], p_control=self.get_p_control_at(stage, k), v=self.V, p=veccat(*self.P), v_control=self.get_v_control_at(stage, k), t=self.integrator_grid[k][i])

    def eval_at_integrator_root(self, stage, expr, k, i, j):
        return stage._expr_apply(expr, t0=self.t0, T=self.T, x=self.xr[k][i][:,j], z=self.zr[k][i][:,j] if self.zk else nan, u=self.U[k], p_control=self.get_p_control_at(stage, k), v=self.V, p=veccat(*self.P), v_control=self.get_v_control_at(stage, k), t=self.tr[k][i][j])

    def set_initial(self, stage, opti, initial):
        for var, expr in initial.items():
            if var is stage.T:
                var = self.T
            if var is stage.t0:
                var = self.t0
            for k in list(range(self.N))+[-1]:
                target = self.eval_at_control(stage, var, k)
                value = DM(opti.debug.value(self.eval_at_control(stage, expr, k), opti.initial()))
                if target.numel()*(self.N)==value.numel():
                    if repmat(target, self.N, 1).shape==value.shape:
                        value = value[k,:]
                    elif repmat(target, 1, self.N).shape==value.shape:
                        value = value[:,k]

                if target.numel()*(self.N+1)==value.numel():
                    if repmat(target, self.N+1, 1).shape==value.shape:
                        value = value[k,:]
                    elif repmat(target, 1, self.N+1).shape==value.shape:
                        value = value[:,k]
                opti.set_initial(target, value)

    def set_value(self, stage, opti, parameter, value):
        for i, p in enumerate(stage.parameters['']):
            if parameter is p:
                opti.set_value(self.P[i], value)
        for i, p in enumerate(stage.parameters['control']):
            if parameter is p:
                opti.set_value(self.P_control[i], value)

    def add_parameter(self, stage, opti):
        for p in stage.parameters['']:
            self.P.append(opti.parameter(p.shape[0], p.shape[1]))
        for p in stage.parameters['control']:
            self.P_control.append(opti.parameter(p.shape[0], self.N * p.shape[1]))

    def set_parameter(self, stage, opti):
        for i, p in enumerate(stage.parameters['']):
            opti.set_value(self.P[i], stage._param_vals[p])
        for i, p in enumerate(stage.parameters['control']):
            opti.set_value(self.P_control[i], stage._param_vals[p])
