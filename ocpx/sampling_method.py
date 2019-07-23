from casadi import integrator, Function, hcat

class SamplingMethod:
  def __init__(self,N=50,M=1,intg='rk'):
    self.N = N
    self.M = M
    self.intg = intg
    # Coefficient matrix from RK4 to reconstruct 4th order polynomial (x0,k1,k2,k3,k4)
    self.rk4_coeff = []
   
  def discrete_system(self,stage):
    f = stage._ode()

    DT = stage.tf/self.N/self.M  # Size of integrator interval
    X0 = f.mx_in("x")            # Initial state
    U = f.mx_in("u")             # Control
    X = X0
    for j in range(self.M):
        k1 = f(X, U)
        k2 = f(X + DT/2 * k1, U)
        k3 = f(X + DT/2 * k2, U)
        k4 = f(X + DT * k3, U)
        self.rk4_coeff.append(hcat([X, k1, k2, k3, k4]))
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    return Function('F', [X0, U], [X,hcat(self.rk4_coeff)],['x0','u'],['xf','rk4_coeff'])    
