from casadi import integrator

class SamplingMethod:
  def __init__(self,N=50,M=1,intg='rk'):
    self.N = N
    self.M = M
    self.intg = intg
   
  def discrete_system(self,stage):
    ode = stage.ode_dict()
    intg = integrator('intg',self.intg,ode,{"t0": 0, "tf": stage.tf/self.N,"number_of_finite_elements": self.M})
    return intg
    
