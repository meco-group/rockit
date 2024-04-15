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

from casadi import integrator, Function, MX, hcat, vertcat, vcat, linspace, veccat, DM, repmat, horzsplit, cumsum, inf, mtimes, symvar, horzcat, symvar, vvcat, is_equal
import casadi as ca
from .direct_method import DirectMethod
from .splines import BSplineBasis, BSpline
from .casadi_helpers import reinterpret_expr, HashOrderedDict, HashDict, is_numeric
from numpy import nan, inf
import numpy as np
from collections import defaultdict
from .splines.micro_spline import bspline_derivative, eval_on_knots, get_greville_points
from .casadi_helpers import get_ranges_dict

# Agnostic about free or fixed start or end point: 

# Agnostic about time?
class Grid:
    def __init__(self, min=0, max=inf):
        self.min = min
        self.max = max

    def __call__(self, t0, T, N):
        return linspace(t0+0.0, t0+T+0.0, N+1)

    def bounds_finalize(self, opti, control_grid, t0_local, tf, N):
        pass

class BSplineSignal:
    def __init__(self, coeff, xi, degree,T=1,parametric=False):
        """
        Parameters
        ----------
        coeff : :obj:`casadi.MX`
            Coefficients of the spline
        xi : :obj:`casadi.MX`
            Grid of the spline (no duplicates)
        degree : int
            Degree of the spline
        T : float, optional
            Time scaling of the spline
            Default: 1
        parametric : bool, optional
            Whether the spline is ony dependant on parameters
        """
        self.coeff = coeff
        self.xi = xi
        self.degree = degree
        [_,self.B] = eval_on_knots(xi, degree)
        self.sampled = horzsplit(coeff @ self.B)
        self.derivative = None
        self.derivative_of = None
        self.T = T
        self.parametric = parametric

        # Initialization delegated to register
        self.peers = None
        self.symbol = None
    
    def sample(self,**kwargs):
        [_,B] = eval_on_knots(self.xi, self.degree, **kwargs)
        return self.coeff @ B

    @property
    def der(self):
        if self.derivative is not None:
            if self.degree==0:
                raise Exception("Cannot differentiate " + self.symbol.name() + " any further.")
            der_symbol = MX.sym("der_"+self.symbol.name(), self.symbol.sparsity())
            self.derivative = self.get_der()
            self.derivative.parametric = self.parametric
            self.derivative.derivative_of = self
            self.peers[der_symbol] = self.derivative
        return self.derivative.symbol

    def get_der(self):
        return BSplineSignal(bspline_derivative(self.coeff,self.xi,self.degree)/self.T, self.xi, self.degree-1, T=self.T)

    @staticmethod
    def register(peers, symbol, stage, signal):
        peers[symbol] = signal
        signal.peers = peers
        signal.symbol = symbol

        # Register derivative if present in stage._signals
        target = stage._signals[symbol]
        if target.derivative is not None:
            signal_der = signal.get_der()
            signal.derivative = signal_der
            signal_der.parametric = signal.parametric
            signal_der.derivative_of = signal
            BSplineSignal.register(peers, target.derivative.symbol, stage, signal_der)

class FixedGrid(Grid):
    def __init__(self, localize_t0=False, localize_T=False, **kwargs):
        Grid.__init__(self, **kwargs)
        self.localize_t0 = localize_t0
        self.localize_T = localize_T

    def bounds_T(self, T_local, t0_local, k, T, N):
        if self.localize_T:
            Tk = T_local[k]
            if k>=0 and k+1<N:
                r = self.constrain_T(T_local[k], T_local[k+1], N)
                if r is not None:
                    yield r
        else:
            Tk = T*(self.normalized(N)[k+1]-self.normalized(N)[k])
        if self.localize_t0 and k>=0:
            yield (t0_local[k]+Tk==t0_local[k+1],{})

    def get_t0_local(self, opti, k, t0, N):
        if self.localize_t0:
            if k==0:
                return t0
            else:
                return opti.variable()
        else:
            return None

    def get_T_local(self, opti, k, T, N):
        if self.localize_T:
            if k==0:
                return T*self.scale_first(N)
            else:
                return opti.variable()
        else:
            return None

class FreeGrid(FixedGrid):
    """Specify a grid with unprescribed spacing
    
    Parameters
    ----------
    min : float or  :obj:`casadi.MX`, optional
        Minimum size of control interval
        Enforced with constraints
        Default: 0
    max : float or  :obj:`casadi.MX`, optional
        Maximum size of control interval
        Enforced with constraints
        Default: inf
    """
    def __init__(self, **kwargs):
        FixedGrid.__init__(self, **kwargs)
        self.localize_T = True

    def constrain_T(self,T,Tnext,N):
        pass

    def bounds_T(self, T_local, t0_local, k, T, N):
        yield (self.min <= (T_local[k] <= self.max),{})
        for i,e in enumerate(FixedGrid.bounds_T(self, T_local, t0_local, k, T, N)):
            yield e

    def bounds_finalize(self, opti, control_grid, t0_local, tf, N):
        opti.subject_to(control_grid[-1]==tf)

    def get_T_local(self, opti, k, T, N):
        return opti.variable()

class UniformGrid(FixedGrid):
    """Specify a grid with uniform spacing
    
    Parameters
    ----------
    min : float or  :obj:`casadi.MX`, optional
        Minimum size of control interval
        Enforced with constraints
        Default: 0
    max : float or  :obj:`casadi.MX`, optional
        Maximum size of control interval
        Enforced with constraints
        Default: inf
    """
    def __init__(self, **kwargs):
        FixedGrid.__init__(self, **kwargs)

    def constrain_T(self,T,Tnext,N):
        return (T==Tnext,{})

    def scale_first(self, N):
        return 1.0/N

    def bounds_T(self, T_local, t0_local, k, T, N):

        for i,e in enumerate(FixedGrid.bounds_T(self, T_local, t0_local, k, T, N)):
            yield e
            if i==0 and k==0:
                if self.localize_T:
                    yield (self.min <= (T_local[0] <= self.max), {})
                else:
                    if self.min==0 and self.max==inf:
                        pass
                    else:
                        yield (self.min <= (T/N <= self.max), {})

    def normalized(self, N):
        return list(np.linspace(0.0, 1.0, N+1))


class FunctionGrid(FixedGrid):
    def __init__(self, normalized_fun, **kwargs):
        """
        Inputs:
            normalized: function that takes N and returns a list of N+1 normalized grid points

        """
        self.normalized_fun = normalized_fun
        FixedGrid.__init__(self, **kwargs)

    def __call__(self, t0, T, N):
        n = self.normalized(N)
        return t0 + hcat(n)*T

    def normalized(self, N):
        return self.normalized_fun(N)
class DensityGrid(FixedGrid):
    def __init__(self, density, integrator='cvodes',integrator_options=None,**kwargs):
        """
        Expression in one symbolic variable (dimensionless time) that describes the density of the grid

        e.g. t**2

        We first compute the definite integral of the density over the interval [0,1]:        
        I = integral_0^t density dt

        
        Next, we inspect the function
        
        E(t) := 1/I*integral_0^t density dt
        
        The grid points t_i are the computed such that E(t_i) is a uniform parition of [0,1]

        """
        self.density = density
        self.t = symvar(density)[0]

        self.integrator_options = {} if integrator_options is None else integrator_options
        self.integrator = integrator
        self.cache = {}

        FixedGrid.__init__(self, **kwargs)

    def __call__(self, t0, T, N):
        n = self.normalized(N)
        return t0 + hcat(n)*T

    def normalized(self, N):
        if N in self.cache: return self.cache[N]
        import scipy
        import scipy.optimize
        x = MX.sym("x")
        scale = MX.sym("scale")
        ode = {"x": x, "ode": scale*ca.substitute(self.density,self.t,self.t*scale), "p": scale, "t":self.t}
        intg = integrator('intg', self.integrator, ode, 0, 1, self.integrator_options)
        I = float(intg(x0=0,p=1)["xf"])
        res = [0]
        for v in list(np.linspace(0.0, 1.0, N+1)[1:-1]*I):
            r = scipy.optimize.root_scalar(lambda tau: float(intg(x0=0,p=tau)["xf"]-v), method='bisect', bracket=[0,1])
            res.append(r.root)
        res.append(1.0)
        self.cache[N] = res
        return res
    
class DenseEdgesGrid(DensityGrid):
    def __init__(self, multiplier=10, edge_frac=0.1, **kwargs):
        interp = ca.interpolant('interp','bspline',[[0.0,edge_frac,1-edge_frac,1.0]],[multiplier,1.0,1.0,multiplier],{"algorithm":"smooth_linear"})
        tau = ca.MX.sym("tau")
        DensityGrid.__init__(self, interp(tau), **kwargs)

class GeometricGrid(FixedGrid):
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
    min : float or  :obj:`casadi.MX`, optional
        Minimum size of control interval
        Enforced with constraints
        Default: 0
    max : float or  :obj:`casadi.MX`, optional
        Maximum size of control interval
        Enforced with constraints
        Default: inf

    Examples
    --------

    >>> MultipleShooting(N=3, grid=GeometricGrid(2)) # grid = [0, 1, 3, 7]*T/7
    >>> MultipleShooting(N=3, grid=GeometricGrid(2,local=True)) # grid = [0, 1, 5, 21]*T/21
    """
    def __init__(self, growth_factor, local=False, **kwargs):
        assert growth_factor>=1
        self._growth_factor = growth_factor
        self.local = local
        FixedGrid.__init__(self, **kwargs)

    def __call__(self, t0, T, N):
        n = self.normalized(N)
        return t0 + hcat(n)*T

    def growth_factor(self, N):
        if not self.local and N>1:
            return self._growth_factor**(1.0/(N-1))
        return self._growth_factor

    def constrain_T(self,T,Tnext,N):
        return (T*self.growth_factor(N)==Tnext,{})

    def scale_first(self, N):
        return self.normalized(N)[1]

    def bounds_T(self, T_local, t0_local, k, T, N):
        if self.localize_T:
            if k==0 or k==-1:
                yield (self.min <= (T_local[k] <= self.max), {})
        else:
            n = self.normalized(N)
            if k==0:
                yield (self.min <= (T*n[1] <= self.max),{})
            if k==-1:
                yield (self.min <= (T*(n[-1]-n[-2]) <= self.max),{})
        for e in FixedGrid.bounds_T(self, T_local, t0_local, k, T, N):
            yield e

    def normalized(self, N):
        g = self.growth_factor(N)
        base = 1.0
        vec = [0]
        for i in range(N):
            vec.append(vec[-1]+base)
            base *= g
        for i in range(N+1):
            vec[i] = vec[i]/vec[-1]
        return vec

class SamplingMethod(DirectMethod):
    def __init__(self, N=50, M=1, intg='rk', intg_options=None, grid=UniformGrid(), **kwargs):
        """
        Parameters
        ----------
        grid : `str` or tuple of :obj:`casadi.MX`
            List of arbitrary expression containing states, controls, ...
        name : `str`
            Name for CasADi Function
        options : dict, optional
            Options for CasADi Function
        intg : `str`
            Rockit-specific methods: 'rk', 'expl_euler' - these allow to make use of signal sampling with 'refine'
            Others are passed on as-is to CasADi
        """
        DirectMethod.__init__(self, **kwargs)
        self.N = N
        self.M = M
        self.intg = intg
        self.intg_options = {} if intg_options is None else intg_options
        self.time_grid = grid
        self.clean()

    def clean(self):
        self.X = []  # List that will hold N+1 decision variables for state vector
        self.Q = []  # List that will hold N+1 expressions for quadrature states
        self.U = []  # List that will hold N decision variables for control vector
        self.Z = []  # Algebraic vars

        self.T = None # Will be set to numbers/opti.variables
        self.t0 = None

        self.P = []
        self.V = None
        self.V_control = []
        self.V_control_plus = []
        self.V_states = []
        self.P_control = []
        self.P_control_plus = []
        self.signals = HashOrderedDict()

        self.poly_coeff = []  # Optional list to save the coefficients for a polynomial
        self.poly_coeff_z = []  # Optional list to save the coefficients for a polynomial
        self.poly_coeff_q = []  # Optional list to save the coefficients for a polynomial
        self.xk = []  # List for intermediate integrator states
        self.xqk = []
        self.zk = []
        self.xr = []
        self.zr = []
        self.tr = []
        self.q = 0

    def discrete_system(self, stage):
        # Coefficient matrix from RK4 to reconstruct 4th order polynomial (k1,k2,k3,k4)
        # nstates x (4 * M)
        poly_coeffs = []
        poly_coeffs_z = []
        poly_coeffs_q = []

        t0 = MX.sym('t0')
        T = MX.sym('T')
        DT = T / self.M

        # Size of integrator interval
        X0 = MX.sym("x", stage.nx)            # Initial state
        U = MX.sym("u", stage.nu)             # Control
        P = MX.sym("p", stage.np+stage.v.shape[0])
        Z = MX.sym("z", stage.nz)

        Z0 = MX.sym("Z0", stage.nz)

        X = [X0]
        Zs = []

        # Compute local start time
        t0_local = t0
        quad = DM.zeros(stage.nxq)
        Q = []

        if stage._state_next:
            intg = stage._diffeq()
        elif hasattr(self, 'intg_' + self.intg):
            intg = getattr(self, "intg_" + self.intg)(stage._ode(), X0, U, P, Z)
        else:
            intg = self.intg_builtin(stage._ode(), X0, U, P, Z)
        if intg.has_free():
            raise Exception("Free variables found: %s" % str(intg.get_free()))
    
        Z0_current = Z0
        for j in range(self.M):
            intg_res = intg(x0=X[-1], u=U, t0=t0_local, DT=DT, DT_control=T, p=P, z0=Z0_current)
            X.append(intg_res["xf"])
            Zs.append(intg_res["zf"])
            poly_coeffs.append(intg_res["poly_coeff"])
            poly_coeffs_z.append(intg_res["poly_coeff_z"])
            poly_coeffs_q.append(intg_res["poly_coeff_q"])
            quad = quad + intg_res["qf"]
            Q.append(quad)
            t0_local += DT
            Z0_current = intg_res["zf"]
        
        ret = Function('F', [X0, U, T, t0, P, Z0], [X[-1], hcat(X), hcat(poly_coeffs), quad, hcat(Q), hcat(poly_coeffs_q), Zs[-1], hcat(Zs), hcat(poly_coeffs_z)],
                       ['x0', 'u', 'T', 't0', 'p', 'z0'], ['xf', 'Xi', 'poly_coeff', 'qf', 'Qi', 'poly_coeff_q', 'zf', 'Zi', 'poly_coeff_z'])
        assert not ret.has_free()
        return ret

    def intg_rk(self, f, X, U, P, Z):
        assert Z.is_empty()
        DT = MX.sym("DT")
        DT_control = MX.sym("DT_control")
        t0 = MX.sym("t0")
        Z0 = MX.sym("z0", 0, 1)
        # A single Runge-Kutta 4 step
        k1 = f(x=X, u=U, p=P, t=t0)
        k2 = f(x=X + DT / 2 * k1["ode"], u=U, p=P, t=t0+DT/2)
        k3 = f(x=X + DT / 2 * k2["ode"], u=U, p=P, t=t0+DT/2)
        k4 = f(x=X + DT * k3["ode"], u=U, p=P, t=t0+DT)

        f0 = k1["ode"]
        f1 = 2/DT*(k2["ode"]-k1["ode"])/2
        f2 = 4/DT**2*(k3["ode"]-k2["ode"])/6
        f3 = 4*(k4["ode"]-2*k3["ode"]+k1["ode"])/DT**3/24
        poly_coeff = hcat([X, f0, f1, f2, f3])

        f0 = k1["quad"]
        f1 = 2/DT*(k2["quad"]-k1["quad"])/2
        f2 = 4/DT**2*(k3["quad"]-k2["quad"])/6
        f3 = 4*(k4["quad"]-2*k3["quad"]+k1["quad"])/DT**3/24
        poly_coeff_q = hcat([f0, f1, f2, f3])

        return Function('F', [X, U, t0, DT, DT_control, P, Z0], [X + DT / 6 * (k1["ode"] + 2 * k2["ode"] + 2 * k3["ode"] + k4["ode"]), poly_coeff, DT / 6 * (k1["quad"] + 2 * k2["quad"] + 2 * k3["quad"] + k4["quad"]), poly_coeff_q, MX(0, 1), MX()], ['x0', 'u', 't0', 'DT', 'DT_control', 'p', 'z0'], ['xf', 'poly_coeff', 'qf', 'poly_coeff_q', 'zf', 'poly_coeff_z'])

    def intg_expl_euler(self, f, X, U, P, Z):
        assert Z.is_empty()
        DT = MX.sym("DT")
        DT_control = MX.sym("DT_control")
        t0 = MX.sym("t0")
        Z0 = MX.sym("z0", 0, 1)
        k = f(x=X, u=U, p=P, t=t0)
        poly_coeff = hcat([X, k["ode"]])
        poly_coeff_q = k["quad"]

        return Function('F', [X, U, t0, DT, DT_control, P, Z0], [X + DT * k["ode"], poly_coeff, DT * k["quad"], poly_coeff_q, MX(0, 1), MX()], ['x0', 'u', 't0', 'DT', 'DT_control', 'p', 'z0'], ['xf', 'poly_coeff', 'qf', 'poly_coeff_q', 'zf', 'poly_coeff_z'])

    def intg_builtin(self, f, X, U, P, Z):
        # A single CVODES step
        DT = MX.sym("DT")
        DT_control = MX.sym("DT_control")
        t = MX.sym("t")
        t0 = MX.sym("t0")
        Z0 = MX.sym("Z0", Z.sparsity())
        res = f(x=X, u=U, p=P, t=t0+t*DT, z=Z)
        data = {'x': X, 'p': vertcat(U, DT, DT_control, P, t0), 'z': Z, 't': t, 'ode': DT * res["ode"], 'quad': DT * res["quad"], 'alg': res["alg"]}
        options = dict(self.intg_options)
        if self.intg in ["collocation"]:
            # In rockit, M replaces number_of_finite_elements on a higher level
            if "number_of_finite_elements" not in options:
                options["number_of_finite_elements"] = 1
        I = integrator('intg_'+self.intg, self.intg, data, options)
        if I.size2_out("xf")!=1:
            raise Exception("Integrator must only return outputs at a single timepoint. Did you specify a grid?")
        res = I.call({'x0': X, 'p': vertcat(U, DT, DT_control, P, t0), 'z0': Z0})
        return Function('F', [X, U, t0, DT, DT_control, P, Z0], [res["xf"], MX(), res["qf"], MX(), res["zf"], MX()], ['x0', 'u', 't0', 'DT', 'DT_control', 'p', 'z0'], ['xf', 'poly_coeff','qf','poly_coeff_q','zf','poly_coeff_z'])

    def untranscribe_placeholders(self, phase, stage):
        pass

    def transcribe_placeholders(self, phase, stage, placeholders):
        """
        Transcription is the process of going from a continuous-time OCP to an NLP
        """
        return stage._transcribe_placeholders(phase, self, placeholders)

    def transcribe_start(self, stage, opti):
        return
        
    def untranscribe(self, stage, phase=1,**kwargs):
        self.clean()

    def transcribe_event_after_varpar(self, stage, phase=1, **kwargs):
        pass

    def transcribe(self, stage, phase=1,**kwargs):
        """
        Transcription is the process of going from a continuous-time OCP to an NLP
        """
        if phase==0: return
        opti = stage.master._method.opti
        if phase==1:

            DM.set_precision(14)

            self.transcribe_start(stage, opti)

            # Grid for B-spline
            self.xi = ca.vec(DM(self.time_grid(0, 1, self.N))).T

            # Parameters needed before variables because of self.T = self.eval(stage, stage._T)
            self.add_parameter(stage, opti)
            self.add_variables(stage, opti)
            self.add_parameter_signals(stage, opti)
            self.set_parameter(stage, opti)

            self.transcribe_event_after_varpar(stage, phase=phase, **kwargs)

            self.integrator_grid = []
            for k in range(self.N):
                t_local = linspace(self.control_grid[k], self.control_grid[k+1], self.M+1)
                self.integrator_grid.append(t_local[:-1] if k<self.N-1 else t_local)
            #self.add_constraints_before(stage, opti)
            self.add_constraints(stage, opti)
            self.add_constraints_after(stage, opti)
            self.add_objective(stage, opti)
        if phase==2:

            self.set_initial(stage, opti, stage._initial)
            T_init = opti.debug.value(self.T, opti.initial())
            t0_init = opti.debug.value(self.t0, opti.initial())

            initial = HashOrderedDict()
            # How to get initial value -> ask opti?
            control_grid_init = self.time_grid(t0_init, T_init, self.N)
            if self.time_grid.localize_t0:
                for k in range(1, self.N):
                    initial[self.t0_local[k]] = control_grid_init[k]
                initial[self.t0_local[self.N]] = control_grid_init[self.N]
            if self.time_grid.localize_T:
                for k in range(not isinstance(self.time_grid, FreeGrid), self.N):
                    initial[self.T_local[k]] = control_grid_init[k+1]-control_grid_init[k]

            self.set_initial(stage, opti, initial)
            self.set_parameter(stage, opti)


    def add_constraints_before(self, stage, opti):
        for c, meta, args in stage._constraints["point"]:
            e = self.eval(stage, c)
            if 'r_at_tf' not in [a.name() for a in symvar(e)]:
                opti.subject_to(e, args["scale"], meta=meta)

    def add_constraints_after(self, stage, opti):
        for c, meta, args in stage._constraints["point"]:
            e = self.eval(stage, c)
            if 'r_at_tf' in [a.name() for a in symvar(e)]:
                opti.subject_to(e, args["scale"], meta=meta)

    def add_inf_constraints(self, stage, opti, c, k, l, meta):
        # Query the discretization method used for polynomial coefficients
        #   interpretation: state ~= coeff * [t^0;t^1;t^2;...]
        #                    t is physical time, but starting at 0 at the beginning of the interval
        coeff = stage._method.poly_coeff[k * self.M + l]

        # Represent polynomial as a BSpline object (https://gitlab.kuleuven.be/meco-software/rockit/-/blob/v0.1.28/rockit/splines/spline.py#L392)
        degree = coeff.shape[1]-1
        basis = BSplineBasis([0]*(degree+1)+[1]*(degree+1),degree)
        tscale = self.T / self.N / self.M
        tpower = vcat([tscale**i for i in range(degree+1)])
        coeff = coeff * repmat(tpower.T,stage.nx,1)
        # TODO: bernstein transformation as function of degree
        Poly_to_Bernstein_matrix_4 = DM([[1,0,0,0,0],[1,1.0/4, 0, 0, 0],[1, 1.0/2, 1.0/6, 0, 0],[1, 3.0/4, 1.0/2, 1.0/4, 0],[1, 1, 1, 1, 1]])
        # Use a direct way to obtain Bernstein coefficients from polynomial coefficients
        state_coeff = mtimes(Poly_to_Bernstein_matrix_4,coeff.T)
        
        # Replace symbols for states by BSpline object derivatives
        subst_from = list(stage.states)

        statesize = [0] + [elem.nnz() for elem in stage.states]
        statessizecum = np.cumsum(statesize)
        state_coeff_split = horzsplit(state_coeff,statessizecum)
        subst_to = [BSpline(basis,coeff) for coeff in state_coeff_split]


        lookup = dict(zip(subst_from, subst_to))

        # Replace symbols for state derivatives by BSpline object derivatives
        subst_from += stage._inf_der.keys()
        dt = (self.control_grid[k + 1] - self.control_grid[k])/self.M
        subst_to += [lookup[e].derivative()*(1/dt) for e in stage._inf_der.values()]

        subst_from += stage._inf_inert.keys()
        subst_to += stage._inf_inert.values()

        # Here, the actual replacement takes place
        c_spline = reinterpret_expr(c, subst_from, subst_to)

        # The use of '>=' or '<=' on BSpline objects automatically results in
        # those same operations being relayed onto BSpline coefficients
        # see https://gitlab.kuleuven.be/meco-software/rockit/-/blob/v0.1.28/rockit/splines/spline.py#L520-521
        try:
            opti.subject_to(self.eval_at_control(stage, c_spline, k), meta=meta)
        except IndexError:
            pass
    def fill_placeholders_integral_control(self, phase, stage, expr, refine=1):
        if phase==1: return
        [ts,exprs] = stage._sample(expr,grid='control',refine=refine)
        return ca.sum2(ca.diff(ts).T*exprs[:,:-1])
        r = 0
        for k in range(self.N):
            dt = self.control_grid[k + 1] - self.control_grid[k]
            r = r + self.eval_at_control(stage, expr, k)*dt
        return r

    def fill_placeholders_sum_control(self, phase, stage, expr, *args):
        if phase==1: return
        r = 0
        for k in range(self.N):
            r = r + self.eval_at_control(stage, expr, k)
        return r

    def fill_placeholders_sum_control_plus(self, phase, stage, expr, *args):
        if phase==1: return
        r = 0
        for k in list(range(self.N))+[-1]:
            r = r + self.eval_at_control(stage, expr, k)
        return r

    def placeholders_next(self, stage, expr, *args):
        self.eval_at_control(stage, expr, k+1)

    def fill_placeholders_at_t0(self, phase, stage, expr, *args):
        if phase==1: return
        return self.eval_at_control(stage, expr, 0)

    def fill_placeholders_at_tf(self, phase, stage, expr, *args):
        if phase==1: return
        return self.eval_at_control(stage, expr, -1)

    def add_objective(self, stage, opti):
        opti.add_objective(self.eval(stage, stage._objective))

    def add_variables_V(self, stage, opti):
        DirectMethod.add_variables(self, stage, opti)

        # Create time grid (might be symbolic)
        self.T = self.eval(stage, stage._T)
        self.t0 = self.eval(stage, stage._t0)

        self.t0_local = [None]*(self.N+1)
        self.T_local = [None]*self.N

        for p in stage.variables['bspline']:
            cat = stage._catalog[p]
            # Compute the degree and size of a BSpline coefficient needed
            d = cat["order"]
            s = self.N+d
            assert p.size2()==1
            C = opti.variable(p.size1(), s)
            BSplineSignal.register(self.signals, p, stage, BSplineSignal(C, self.xi, d, T=self.T))


    def add_variables_V_control(self, stage, opti, k):
        if k==0:
            self.V_control = [[] for v in stage.variables['control']]
            self.V_control_plus = [[] for v in stage.variables['control+']]
            for i, v in enumerate(stage.variables['states']):
                self.V_states.append([opti.variable(v.shape[0], v.shape[1], scale=stage._scale[v])])
        for i, v in enumerate(stage.variables['control']):
            self.V_control[i].append(opti.variable(v.shape[0], v.shape[1], scale=stage._scale[v]))
        for i, v in enumerate(stage.variables['control+']):
            self.V_control_plus[i].append(opti.variable(v.shape[0], v.shape[1], scale=stage._scale[v]))
        for i, v in enumerate(stage.variables['states']):
            self.V_states[i].append(opti.variable(v.shape[0], v.shape[1], scale=stage._scale[v]))

        self.t0_local[k] = self.time_grid.get_t0_local(opti, k, self.t0, self.N)
        self.T_local[k] = self.time_grid.get_T_local(opti, k, self.T, self.N)

    def add_variables_V_control_finalize(self, stage, opti):
        for i, v in enumerate(stage.variables['control+']):
            self.V_control_plus[i].append(opti.variable(v.shape[0], v.shape[1], scale=stage._scale[v]))
        if self.time_grid.localize_t0:
            self.t0_local[self.N] = opti.variable()
            self.control_grid = hcat(self.t0_local)

        elif self.time_grid.localize_T:
            t0 = self.t0
            cumsum = [t0]
            for e in self.T_local:
                cumsum.append(cumsum[-1]+e)
            self.control_grid = hcat(cumsum)
        else:
            self.control_grid = self.time_grid(self.t0, self.T, self.N)

        self.time_grid.bounds_finalize(opti, self.control_grid, self.t0_local, self.t0+self.T, self.N)

    def add_coupling_constraints(self, stage, opti, k):
        advanced = opti.advanced
        for c,kwargs in self.time_grid.bounds_T(self.T_local, self.t0_local, k, self.T, self.N):
            if not advanced.is_parametric(c):
                opti.subject_to(c,**kwargs)

    def get_p_control_at(self, stage, k=-1):
        return veccat(*[p[k] for p in self.P_control])

    def get_v_control_at(self, stage, k=-1):
        return veccat(*[v[k] for v in self.V_control])

    def get_p_control_plus_at(self, stage, k=-1):
        return veccat(*[p[k] for p in self.P_control_plus])

    def get_v_control_plus_at(self, stage, k=-1):
        return veccat(*[v[k] for v in self.V_control_plus])

    def get_v_states_at(self, stage, k=-1):
        return veccat(*[v[k] for v in self.V_states])

    def get_signals_at(self, stage, k=-1):
        return veccat(*[e.sampled[k] for e in self.signals.values()])

    def get_p_sys(self, stage, k, include_signals=True):
        args = [vvcat(self.P),
                self.get_p_control_at(stage, k),
                self.get_p_control_plus_at(stage, k),
                self.V, self.get_v_control_at(stage, k),
                self.get_v_control_plus_at(stage, k)]
        if include_signals:
            args.append(self.get_signals_at(stage, k))
        return vcat(args)

    def eval(self, stage, expr):
        return stage.master._method.eval_top(stage.master,
                                             stage._expr_apply(expr,
                                                               p=veccat(*self.P),
                                                               v=self.V,
                                                               t0=stage.t0,
                                                               T=stage.T))

    def eval_at_control(self, stage, expr, k):
        try:
            syms = symvar(expr)
        except:
            syms = []
        offsets = defaultdict(list)
        symbols = defaultdict(list)
        for s in syms:
            if s in stage._offsets:
                e, offset = stage._offsets[s]
                offsets[offset].append(e)
                symbols[offset].append(s)

        subst_from = []
        subst_to = []
        for offset in offsets.keys():
            if k==-1 and offset>0:
                raise IndexError()
            if k+offset<0:
                raise IndexError()
            subst_from.append(vvcat(symbols[offset]))
            subst_to.append(self._eval_at_control(stage, vvcat(offsets[offset]), k+offset))
            #print(expr, subst_from, subst_to)


        DT_control = self.get_DT_control_at(k)
        DT = self.get_DT_at(k, self.M-1 if k==-1 else 0)
        if self.Q:
            xq = self.Q[k]
        else:
            if k==-1:
                xq = self.q
            else:
                xq = nan
        expr = stage._expr_apply(expr,
                                 sub=(subst_from, subst_to),
                                 t0=self.t0,
                                 T=self.T,
                                 x=self.X[k],
                                 z=self.Z[k] if self.Z else nan,
                                 xq=xq,
                                 u=self.U[k],
                                 p_control=self.get_p_control_at(stage, k),
                                 p_control_plus=self.get_p_control_plus_at(stage, k),
                                 v=self.V, p=veccat(*self.P),
                                 v_control=self.get_v_control_at(stage, k),
                                 v_control_plus=self.get_v_control_plus_at(stage, k),
                                 signals=(self.signals, self.get_signals_at(stage, k)),
                                 v_states=self.get_v_states_at(stage, k),
                                 t=self.control_grid[k],
                                 DT=DT,
                                 DT_control=DT_control)
        expr = stage.master._method.eval_top(stage.master, expr)
        return expr
    
    def get_DT_control_at(self, k):
        if k==-1 or k==self.N:
            return self.control_grid[-1]-self.control_grid[-2]
        return self.control_grid[k+1]-self.control_grid[k]

    def get_DT_at(self, k, i):
        integrator_grid = self.integrator_grid[k]
        if i<integrator_grid.numel()-1:
            return integrator_grid[i+1]-integrator_grid[i]
        else:
            return self.integrator_grid[k+1][0]-self.integrator_grid[k][i]

    def _eval_at_control(self, stage, expr, k):
        x = self.X[k]
        if self.Q:
            xq = self.Q[k]
        else:
            if k==-1:
                xq = self.q
            else:
                xq = nan
        z = self.Z[k] if self.Z else nan
        u = self.U[-1] if k==len(self.U) else self.U[k] # Would be fixed if we remove the (k=-1 case)
        p_control = self.get_p_control_at(stage, k) if k!=len(self.U) else self.get_p_control_at(stage, k-1)
        v_control = self.get_v_control_at(stage, k) if k!=len(self.U) else self.get_v_control_at(stage, k-1)
        p_control_plus = self.get_p_control_plus_at(stage, k)
        v_control_plus = self.get_v_control_plus_at(stage, k)
        v_states = self.get_v_states_at(stage, k)
        t = self.control_grid[k]
        DT_control = self.get_DT_control_at(k)
        if k==-1 or k==len(self.integrator_grid):
            DT = self.get_DT_at(len(self.integrator_grid)-1, self.M-1)
        else:
            DT = self.get_DT_at(k, 0)
        return stage._expr_apply(expr,
                                 t0=self.t0,
                                 T=self.T,
                                 x=x,
                                 z=z,
                                 xq=xq,
                                 u=u,
                                 p_control=p_control,
                                 p_control_plus=p_control_plus,
                                 v=self.V, p=veccat(*self.P),
                                 v_control=v_control,
                                 v_control_plus=v_control_plus,
                                 v_states=v_states,
                                 t=t,
                                 DT=DT,
                                 DT_control=DT_control)

    def eval_at_integrator(self, stage, expr, k, i):
        DT_control = self.get_DT_control_at(k)
        DT = self.get_DT_at(k, i)
        return stage.master._method.eval_top(stage.master,
                                             stage._expr_apply(expr,
                                                               t0=self.t0,
                                                               T=self.T,
                                                               x=self.xk[k*self.M + i],
                                                               xq=self.xqk[k*self.M + i],
                                                               z=self.zk[k*self.M + i] if self.zk else nan,
                                                               u=self.U[k], p_control=self.get_p_control_at(stage, k),
                                                               p_control_plus=self.get_p_control_plus_at(stage, k),
                                                               v=self.V, p=veccat(*self.P),
                                                               v_control=self.get_v_control_at(stage, k),
                                                               v_control_plus=self.get_v_control_plus_at(stage, k),
                                                               v_states=self.get_v_states_at(stage, k),
                                                               t=self.integrator_grid[k][i],
                                                               DT=DT,
                                                               DT_control=DT_control))

    def eval_at_integrator_root(self, stage, expr, k, i, j):
        DT_control = self.get_DT_control_at(k)
        DT = self.get_DT_at(k, i)
        return stage.master._method.eval_top(stage.master,
                                             stage._expr_apply(expr,
                                                               t0=self.t0,
                                                               T=self.T,
                                                               x=self.xr[k][i][:,j],
                                                               z=self.zr[k][i][:,j] if self.zk else nan,
                                                               u=self.U[k],
                                                               p_control=self.get_p_control_at(stage, k),
                                                               p_control_plus=self.get_p_control_plus_at(stage, k),
                                                               v=self.V, p=veccat(*self.P),
                                                               v_control=self.get_v_control_at(stage, k),
                                                               v_control_plus=self.get_v_control_plus_at(stage, k),
                                                               t=self.tr[k][i][j],
                                                               DT=DT,
                                                               DT_control=DT_control))

    def set_initial(self, stage, master, initial):
        opti = master.opti if hasattr(master, 'opti') else master
        opti.cache_advanced()
        initial = HashOrderedDict(initial)
        algs = get_ranges_dict(stage.algebraics)
        initial_alg = HashDict()
        for a, v in list(initial.items()):
            if a in algs:
                initial_alg[a] = v
                del initial[a]
        for var, expr in initial.items():
            if is_equal(var, stage.T):
                var = self.T
            if is_equal(var, stage.t0):
                var = self.t0
            opti_initial = opti.initial()
            if is_numeric(expr):
                value = ca.evalf(expr)
            else:
                expr = ca.hcat([self.eval_at_control(stage, expr, k) for k in list(range(self.N))+[-1]]) # HOT line
                value = DM(opti.debug.value(expr, opti_initial))
            # Row vector if vector
            if value.is_column() and var.is_scalar(): value = value.T
            if var in self.signals:
                target = stage.sample(var,'gist')[1]
                opti.set_initial(target, ca.repmat(value,1,target.shape[1]), cache_advanced=True)
            for k in list(range(self.N))+[-1]:
                target = self.eval_at_control(stage, var, k)
                value_k = value
                if target.numel()*(self.N)==value.numel() or target.numel()*(self.N+1)==value.numel():
                    value_k = value[:,k]
                try:
                    #print(target,value_k)
                    opti.set_initial(target, value_k, cache_advanced=True)
                except Exception as e:
                    # E.g for single shooting, set_initial of a state, for k>0
                    # Error message is usually "... arbitrary expression ..." but can also be
                    # "... You cannot set an initial value for a parameter ..."
                    # if the dynamics contains a parameter
                    if "arbitrary expression" in str(e) or (not target.is_valid_input() and "initial value for a parameter" in str(e)):
                        pass
                    else:
                        # Other type of error: 
                        raise e
        for var, expr in initial_alg.items():
            opti_initial = opti.initial()
            if is_numeric(expr):
                value = ca.evalf(expr)
            else:
                expr = ca.hcat([self.eval_at_control(stage, expr, k) for k in range(self.N)]) # HOT line
                value = DM(opti.debug.value(expr, opti_initial))

            # Row vector if vector
            if value.is_column() and var.is_scalar(): value = value.T


            print("value",value)
            for k in range(self.N):
                value_k = value
                z0 = self.Z0[k]
                if z0 is None: break
                target = z0[algs[var]]
                if target.numel()*(self.N)==value.numel() or target.numel()*(self.N+1)==value.numel():
                    value_k = value[:,k]
                opti.set_value(target, value_k)

    def set_value(self, stage, master, parameter, value):
        opti = master.opti if hasattr(master, 'opti') else master
        found = False
        for i, p in enumerate(stage.parameters['']):
            if is_equal(parameter, p):
                found = True
                opti.set_value(self.P[i], value)
        for i, p in enumerate(stage.parameters['control']):
            if is_equal(parameter, p):
                found = True
                opti.set_value(hcat(self.P_control[i]), value)
        for i, p in enumerate(stage.parameters['control+']):
            if is_equal(parameter, p):
                found = True
                opti.set_value(hcat(self.P_control_plus[i]), value)
        for p in stage.parameters['bspline']:
            if is_equal(parameter, p):
                found = True
                opti.set_value(self.signals[p].coeff, value)
        assert found, "You attempted to set the value of a non-parameter."
    def add_parameter(self, stage, opti):
        for p in stage.parameters['']:
            self.P.append(opti.parameter(p.shape[0], p.shape[1]))
        for p in stage.parameters['control']:
            self.P_control.append([opti.parameter(p.shape[0], p.shape[1]) for i in range(self.N)])
        for p in stage.parameters['control+']:
            self.P_control_plus.append([opti.parameter(p.shape[0], p.shape[1]) for i in range(self.N+1)])

    def add_parameter_signals(self, stage, opti):
        # These in general depend on self.T (at least derivatives), which is not yet available in add_parameter
        for p in stage.parameters["bspline"]:
            cat = stage._catalog[p]
            # Compute the degree and size of a BSpline coefficient needed
            d = cat["order"]
            s = self.N+d
            assert p.size2()==1
            C = opti.parameter(p.size1(), s)
            BSplineSignal.register(self.signals, p, stage, BSplineSignal(C, self.xi, d, T = self.T,parametric=True))

    def set_parameter(self, stage, opti):
        for i, p in enumerate(stage.parameters['']):
            opti.set_value(self.P[i], stage._param_value(p))
        for i, p in enumerate(stage.parameters['control']):
            opti.set_value(hcat(self.P_control[i]), stage._param_value(p))
        for i, p in enumerate(stage.parameters['control+']):
            opti.set_value(hcat(self.P_control_plus[i]), stage._param_value(p))
        for p in stage.parameters['bspline']:
            opti.set_value(self.signals[p].coeff, stage._param_value(p))
