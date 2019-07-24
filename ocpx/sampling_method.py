from casadi import integrator, Function, MX, hcat, vcat

class SamplingMethod:
  def __init__(self,N=50,M=1,intg='rk4'):
    self.N = N
    self.M = M
    self.intg = intg
    # Coefficient matrix from RK4 to reconstruct 4th order polynomial (x0,k1,k2,k3,k4)
    self.poly_coeff = []

  def discrete_system(self,stage):
    f = stage._ode()

    T = stage.T
    DT = T/self.N/self.M

    # Size of integrator interval
    X0 = f.mx_in("x")            # Initial state
    U = f.mx_in("u")             # Control
    P = f.mx_in("p")
    X = [X0]
    intg = getattr(self, "intg_"+self.intg)(f,X0,DT,U,P)
    for j in range(self.M):
      X.append(intg(X[-1],U,P))

    return Function('F', [X0, U, P], [X[-1], hcat(X), hcat(self.poly_coeff)], ['x0','u', 'p'], ['xf', 'Xi', 'poly_coeff'])

  def intg_rk(self,f,X,DT,U,P):
    # A single Runge-Kutta 4 step
    k1 = f(X, U, P)
    k2 = f(X + DT/2 * k1, U, P)
    k3 = f(X + DT/2 * k2, U, P)
    k4 = f(X + DT * k3, U, P)
    self.poly_coeff.append(hcat([X, k1,k2,k3,k4]))
    return Function('F', [X, U, P], [X + DT/6*(k1 +2*k2 +2*k3 +k4)], ['x0','u', 'p'], ['xf'])

  def intg_cvodes(self,f,X,DT,U,P):
    # A single CVODES step
    data, opts = self.prepare_sundials(f,X,DT,U,P)
    I = integrator('intg_cvodes', 'cvodes', data, opts)

    return Function('F', [X, U,P], [I.call({'x0': X, 'p': vcat([U, P])})['xf']], ['x0','u','p'],['xf'])

  def intg_idas(self,f,X,DT,U,P):
    # A single IDAS step
    data, opts = self.prepare_sundials(f,X,DT,U,P)
    I = integrator('intg_idas', 'idas', data, opts)

    return Function('F', [X, U, P], [I.call({'x0': X, 'p': vcat([U, P])})['xf']], ['x0','u','p'],['xf'])

  def prepare_sundials(self,f,X,DT,U,P):
    # Preparing arguments of Sundials integrators
    opts = {} # TODO - additional options
    opts['tf'] = 1
    data = {'x': X, 'p': vcat([U, P]), 'ode': DT*f(X,U,P)}
    # data = {'x': X, 't',t, 'p': U, 'ode': substitute(DT*f(X,U),t,t*DT)}

    return (data, opts)
