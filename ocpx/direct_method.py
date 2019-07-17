from casadi import Opti

class DirectMethod:
  """
  Base class for 'direct' solution methods for Optimal Control Problems:
    'first discretize, then optimize'
  """
  def __init__(self,solver):
    self.opti = Opti()
    self.opti.solver(solver)