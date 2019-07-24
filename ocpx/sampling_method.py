from casadi import integrator, Function, MX, hcat

class SamplingMethod:
  def __init__(self,N=50,M=1,intg='rk'):
    self.N = N
    self.M = M
    self.intg = intg

  def discrete_system(self,stage):
    f = stage._ode()

    # intermediate integrator states should result in a (nstates x M)
    xk = []
    # Coefficient matrix from RK4 to reconstruct 4th order polynomial (k1,k2,k3,k4)
    # nstates x (4 * M)
    poly_coeffs = []

    T = MX.sym('T')
    DT = T/self.N/self.M

    # Size of integrator interval
    X0 = f.mx_in("x")            # Initial state
    U = f.mx_in("u")             # Control

    X = [X0]
    intg = getattr(self, "intg_"+self.intg)(f,X0,DT,U)
    for j in range(self.M):
      intg_res = intg(x0=X[-1],u=U)
      X.append(intg_res["xf"])
      poly_coeffs.append(intg_res["poly_coeff"])

    return Function('F', [X0, U, T], [X[-1], hcat(X), hcat(poly_coeffs)],
                    ['x0','u', 'T'], ['xf', 'Xi', 'poly_coeff'])



  def intg_rk(self,f,X,DT,U):
    # A single Runge-Kutta 4 step
    k1 = f(X, U)
    k2 = f(X + DT/2 * k1, U)
    k3 = f(X + DT/2 * k2, U)
    k4 = f(X + DT * k3, U)
    poly_coeff = hcat([X, k1,k2,k3,k4])
    return Function('F', [X, U], [X + DT/6*(k1 +2*k2 +2*k3 +k4),poly_coeff], ['x0','u'], ['xf','poly_coeff'])

  def intg_cvodes(self,f,X,DT,U):
    # A single CVODES step
    data, opts = self.prepare_sundials(f,X,DT,U)
    I = integrator('intg_cvodes', 'cvodes', data, opts)

    return Function('F', [X, U], [I.call({'x0': X, 'p': U})['xf'],MX()], ['x0','u'],['xf','poly_coeff'])

  def intg_idas(self,f,X,DT,U):
    # A single IDAS step
    data, opts = self.prepare_sundials(f,X,DT,U)
    I = integrator('intg_idas', 'idas', data, opts)

    return Function('F', [X, U], [I.call({'x0': X, 'p': U})['xf'],MX()], ['x0','u'],['xf','poly_coeff'])

  def prepare_sundials(self,f,X,DT,U):
    # Preparing arguments of Sundials integrators
    opts = {} # TODO - additional options
    opts['tf'] = 1
    data = {'x': X, 'p': U, 'ode': DT*f(X,U)}
    # data = {'x': X, 't',t, 'p': U, 'ode': substitute(DT*f(X,U),t,t*DT)}

    return (data, opts)