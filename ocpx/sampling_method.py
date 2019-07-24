from casadi import integrator, Function, MX, hcat

class SamplingMethod:
  def __init__(self,N=50,M=1,intg='rk'):
    self.N = N
    self.M = M
    self.intg = intg
    # Coefficient matrix from RK4 to reconstruct 4th order polynomial (x0,k1,k2,k3,k4)
    self.rk4_coeff = []
   
  def discrete_system(self,stage):
    f = stage._ode()

      # Size of integrator interval
    X0 = f.mx_in("x")            # Initial state
    U = f.mx_in("u")             # Control
    X = X0
    t0 = MX.sym("t0")
    tf = MX.sym("tf")
    DT = (tf-t0)/self.M
    for j in range(self.M):
        k1 = f(X, U)
        k2 = f(X + DT/2 * k1, U)
        k3 = f(X + DT/2 * k2, U)
        k4 = f(X + DT * k3, U)
        self.rk4_coeff.append(hcat([X, k1, k2, k3, k4]))
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    return Function('F', [X0, U, t0, tf], [X,hcat(self.rk4_coeff)],['x0','u','t0','tf'],['xf','rk4_coeff'])
