from casadi import MX, substitute, Function, vcat, depends_on, vertcat, jacobian, veccat
from .freetime import FreeTime
from .stage_options import GridControl, GridIntegrator


class Stage:
    def __init__(self, ocp, t0=0, T=1):
        self.ocp = ocp
        self.states = []
        self.controls = []
        self.parameters = []
        self._param_vals = dict()
        self._state_der = dict()
        self._constraints = []
        self._expr_t0 = dict()  # Expressions defined at t0
        self._expr_tf = dict()  # Expressions defined at tf
        self._objective = 0
        self._initial = dict()
        self._T = T
        self._t0 = t0
        self._create_variables(t0, T)

    def _create_variables(self, t0, T):
        if self.is_free_time():
            self.T = MX.sym('T')
        else:
            self.T = T

        if self.is_free_starttime():
            self.t0 = MX.sym('t0')
        else:
            self.t0 = t0

        self.tf = self.t0 + self.T

        self.t = MX.sym('t')

    def is_free_time(self):
        return isinstance(self._T, FreeTime)

    def is_free_starttime(self):
        return isinstance(self._t0, FreeTime)

    def get_jacobian(self, der, state):
        return jacobian(der, state)

    def state(self, dimm=1, dimn=1):
        """
        Create a state
        """
        # Create a placeholder symbol with a dummy name (see #25)
        x = MX.sym("x", dimm, dimn)
        self.states.append(x)
        return x

    def parameter(self, dim=1):
        """
        Create a parameter
        """
        # Create a placeholder symbol with a dummy name (see #25)
        p = MX.sym("p", dim)
        self.parameters.append(p)
        return p

    def control(self, dimm=1, dimn=1, order=0):
        if order >= 1:
            u = self.state(dimm, dimn)
            helper_u = self.control(dimm=dimm, dimn=dimn, order=order - 1)
            self.set_der(u, helper_u)
            return u

        u = MX.sym("u", dimm, dimn)
        self.controls.append(u)
        return u

    def set_value(self, parameter, value):
        self._param_vals[parameter] = value

    def set_der(self, state, der):
        self._state_der[state] = der

    def integral(self, expr):
        I = self.state()
        self.set_der(I, expr)
        self.subject_to(self.at_t0(I) == 0)
        return self.at_tf(I)

    def subject_to(self, constr):
        self._constraints.append(constr)

    def set_initial(self, var, expr):
        self._initial[var] = expr

    def at_t0(self, expr):
        p = MX.sym("p_t0", expr.sparsity())
        self._expr_t0[p] = expr
        return p

    def at_tf(self, expr):
        p = MX.sym("p_tf", expr.sparsity())
        self._expr_tf[p] = expr
        return p

    def add_objective(self, term):
        self._objective = self._objective + term

    def method(self, method):
        self._method = method

    @property
    def x(self):
        return veccat(*self.states)

    @property
    def u(self):
        return vcat(self.controls)

    @property
    def p(self):
        return vcat(self.parameters)

    @property
    def nx(self):
        return self.x.numel()

    @property
    def nu(self):
        return self.u.numel()

    @property
    def np(self):
        return self.p.numel()

    def is_trajectory(self, expr):
        return depends_on(expr, vertcat(self.x, self.u))

    # Internal methods
    def _ode(self):
        ode = veccat(*[self._state_der[k] for k in self.states])
        return Function('ode', [self.x, self.u, self.p], [ode], ["x", "u", "p"], ["ode"])

    def _bake(self, x0=None, xf=None, u0=None, uf=None):
        for k in self._expr_t0.keys():
            self._expr_t0[k] = substitute(
                [self._expr_t0[k]], [self.x, self.u], [x0, u0])[0]
        for k in self._expr_tf.keys():
            self._expr_tf[k] = substitute(
                [self._expr_tf[k]], [self.x, self.u], [xf, uf])[0]

    def _boundary_constraints_expr(self):
        return [c for c in self._constraints if not self.is_trajectory(c)]

    def _path_constraints_expr(self):
        return [c for c in self._constraints if self.is_trajectory(c)]

    def _expr_apply(self, expr, **kwargs):
        """
        Substitute placeholder symbols with actual decision variables,
        or expressions involving decision variables
        """
        subst_from, subst_to = self.get_subst_set(**kwargs)
        return substitute([expr], subst_from, subst_to)[0]

    def get_subst_set(self, **kwargs):
        subst_from = []
        subst_to = []
        for k, v in self._expr_t0.items():
            subst_from.append(k)
            subst_to.append(v)
        for k, v in self._expr_tf.items():
            subst_from.append(k)
            subst_to.append(v)
        if "t" in kwargs:
            subst_from.append(self.t)
            subst_to.append(kwargs["t"])
        if "x" in kwargs:
            subst_from.append(self.x)
            subst_to.append(kwargs["x"])
        if "u" in kwargs:
            subst_from.append(self.u)
            subst_to.append(kwargs["u"])
        if self.is_free_starttime() and "t0" in kwargs:
            subst_from.append(self.t0)
            subst_to.append(kwargs["t0"])

        if self.is_free_time() and "T" in kwargs:
            subst_from.append(self.T)
            subst_to.append(kwargs["T"])

        if "p" in kwargs:
            for i, p in enumerate(self.parameters):
                subst_from.append(p)
                subst_to.append(kwargs["p"][i])

        return (subst_from, subst_to)

    def subst_expr(self, expr):
        for k in range(self._method.N):
            subst_from, subst_to = self.get_subst_set(
                x=self._method.X[k],
                u=self._method.U[k],
                T=self._method.T,
                p=self._method.P,
                t0=self._method.t0,
            )
            expr = substitute([expr], subst_from, subst_to)[0]

        return expr

    _constr_apply = _expr_apply

    def _expr_to_function(self, expr):
        return Function('helper', [self.x, self.u], [expr], ["x", "u"], ["out"])

    @property
    def grid_control(self):
        return GridControl()

    @property
    def grid_integrator(self):
        return GridIntegrator()
