from casadi import integrator, Function, MX, hcat, vertcat



class SamplingMethod:
    def __init__(self, N=50, M=1, intg='rk'):
        self.N = N
        self.M = M
        self.intg = intg

    def discrete_system(self, stage):
        f = stage._ode()

        # intermediate integrator states should result in a (nstates x M)
        xk = []
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
