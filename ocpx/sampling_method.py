from casadi import integrator, Function, MX, hcat, vertcat, vcat, linspace, veccat
from .direct_method import DirectMethod

class SamplingMethod(DirectMethod):
    def __init__(self, N=50, M=1, intg='rk', **kwargs):
        DirectMethod.__init__(self, **kwargs)
        self.N = N
        self.M = M
        self.intg = intg

        self.X = []  # List that will hold N+1 decision variables for state vector
        self.U = []  # List that will hold N decision variables for control vector
        self.T = None
        self.t0 = None
        self.P = []
        self.V = None
        self.V_control = []
        self.P_control = []

        self.poly_coeff = None  # Optional list to save the coefficients for a polynomial
        self.xk = []  # List for intermediate integrator states

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

        X = [X0]
        intg = getattr(self, "intg_" + self.intg)(f, X0, U, P)
        assert not intg.has_free()
        for j in range(self.M):
            intg_res = intg(x0=X[-1], u=U, DT=DT, p=P)
            X.append(intg_res["xf"])
            poly_coeffs.append(intg_res["poly_coeff"])

        ret = Function('F', [X0, U, T, t0, P], [X[-1], hcat(X), hcat(poly_coeffs)],
                       ['x0', 'u', 'T', 't0', 'p'], ['xf', 'Xi', 'poly_coeff'])
        assert not ret.has_free()
        return ret

    def intg_rk(self, f, X, U, P):
        DT = MX.sym("DT")
        # A single Runge-Kutta 4 step
        k1 = f(X, U, P)
        k2 = f(X + DT / 2 * k1, U, P)
        k3 = f(X + DT / 2 * k2, U, P)
        k4 = f(X + DT * k3, U, P)

        f0 = k1
        f1 = 2/DT*(k2-k1)/2
        f2 = 4/DT**2*(k3-k2)/6
        f3 = 4*(k4-2*k3+k1)/DT**3/24
        poly_coeff = hcat([X, f0, f1, f2, f3])
        return Function('F', [X, U, DT, P], [X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), poly_coeff], ['x0', 'u', 'DT', 'p'], ['xf', 'poly_coeff'])

    def intg_cvodes(self, f, X, U, P):
        # A single CVODES step
        data, opts = self.prepare_sundials(f, X, U, P)
        I = integrator('intg_cvodes', 'cvodes', data, opts)
        DT = MX.sym("DT")
        return Function('F', [X, U, DT, P], [I.call({'x0': X, 'p': vertcat(U, DT, P)})['xf'], MX()], ['x0', 'u', 'DT', 'p'], ['xf', 'poly_coeff'])

    def intg_idas(self, f, X, U, P):
        # A single IDAS step
        data, opts = self.prepare_sundials(f, X, U, P)
        I = integrator('intg_idas', 'idas', data, opts)
        DT = MX.sym("DT")

        return Function('F', [X, U, DT, P], [I.call({'x0': X, 'p': vertcat(U, DT, P)})['xf'], MX()], ['x0', 'u', 'DT', 'p'], ['xf', 'poly_coeff'])

    def prepare_sundials(self, f, X, U, P):
        # Preparing arguments of Sundials integrators
        opts = {}  # TODO - additional options
        opts['tf'] = 1
        DT = MX.sym("DT")
        data = {'x': X, 'p': vertcat(U, DT, P), 'ode': DT * f(X, U, P)}
        # data = {'x': X, 't',t, 'p': U, 'ode': substitute(DT*f(X,U),t,t*DT)}

        return (data, opts)


    def transcribe(self, stage, opti):
        """
        Transcription is the process of going from a continous-time OCP to an NLP
        """
        self.add_variables(stage, opti)
        self.add_parameter(stage, opti)

        # Create time grid (might be symbolic)
        self.control_grid = linspace(MX(self.t0), self.t0 + self.T, self.N + 1)
        placeholders = stage._bake_placeholders(self)
        self.add_constraints(stage, opti)
        self.add_objective(stage, opti)
        self.set_initial(stage, opti)
        self.set_parameter(stage, opti)
        return placeholders

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

    def add_time_variables(self, stage, opti):
        if stage.is_free_time():
            self.T = opti.variable()
            opti.set_initial(self.T, stage._T.T_init)
        else:
            self.T = stage.T

        if stage.is_free_starttime():
            self.t0 = opti.variable()
            opti.set_initial(self.t0, stage._t0.T_init)
        else:
            self.t0 = stage.t0

    def get_p_control_at(self, stage, k=-1):
        return veccat(*[p[:,k] for p in self.P_control])

    def get_v_control_at(self, stage, k=-1):
        return veccat(*[v[k] for v in self.V_control])

    def eval(self, stage, expr):
        return stage._expr_apply(expr, p=veccat(*self.P), v=self.V)

    def eval_at_control(self, stage, expr, k):
        return stage._expr_apply(expr, x=self.X[k], u=self.U[k], p_control=self.get_p_control_at(stage, k), v=self.V, p=veccat(*self.P), v_control=self.get_v_control_at(stage, k), t=self.control_grid[k])

    def set_initial(self, stage, opti):
        for var, expr in stage._initial.items():
            for k in range(self.N):
                opti.set_initial(
                    self.eval_at_control(stage, var, k),
                    opti.debug.value(self.eval_at_control(stage, expr, k), opti.initial()))
            opti.set_initial(
                self.eval_at_control(stage, var, -1),
                opti.debug.value(self.eval_at_control(stage, expr, -1), opti.initial()))

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
            try:
                opti.set_value(self.P[i], stage._param_vals[p])
            except:
                import ipdb
                ipdb._set_trace()
        for i, p in enumerate(stage.parameters['control']):
            opti.set_value(self.P_control[i], stage._param_vals[p])
