from casadi import integrator, Function

class SamplingMethod:
  def __init__(self,N=50,M=1,intg='rk'):
    self.N = N
    self.M = M
    self.intg = intg

  def discrete_system(self,stage):
    f = stage._ode()

    DT = stage.tf/self.N/self.M  # Size of integrator interval
    X0 = f.mx_in("x")            # Initial state
    U = f.mx_in("u")             # Control

    return getattr(self, "intg_"+self.intg)(f,X0,DT,U)

  def intg_rk(self,f,X0,DT,U):
    X = X0
    for j in range(self.M):
        k1 = f(X, U)
        k2 = f(X + DT/2 * k1, U)
        k3 = f(X + DT/2 * k2, U)
        k4 = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    return Function('F', [X0, U], [X], ['x0','u'], ['xf'])

  def intg_cvodes(self,f,X0,DT,U):
    opts = {} # TODO - additional options
    opts['tf'] = DT
    X = X0
    _f = {'x': X, 'p': U, 'ode': f(X,U)}
    I = integrator('blabla', 'cvodes', _f, opts)

    return Function('F', [X0, U], I(X, U, [], [], [], []), ['x0','u'],['xf','qf','zf','rxf','rqf','rzf'])
