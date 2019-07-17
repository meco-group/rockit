from .sampling_method import SamplingMethod

class MultipleShooting(SamplingMethod):
  def __init__(self,*args,**kwargs):
    SamplingMethod.__init__(self,*args,**kwargs)
    self.state_variables = []
    self.control_variables = []

  def transcribe(self,stage,opti):
    self.add_variables(stage,opti)
    stage._bake(x0=self.state_variables[0],
               xf=self.state_variables[-1],
               u0=self.control_variables[0],
               uf=self.control_variables[-1])
    self.add_constraints(stage,opti)
    self.add_objective(stage,opti)

  def add_variables(self,stage,opti):
    self.state_variables.append(opti.variable(stage.nx))

    for k in range(self.N):
      self.control_variables.append(opti.variable(stage.nu))
      self.state_variables.append(opti.variable(stage.nx))

  def add_constraints(self,stage,opti):
    F = self.discrete_system(stage)

    for k in range(self.N):
      xk = self.state_variables[k]
      uk = self.control_variables[k]
      xk_next = self.state_variables[k+1]
      opti.subject_to(xk_next==F(x0=xk,p=uk)["xf"])

      for c in stage._path_constraints_expr():
        opti.subject_to(stage._expr_apply(c,x=xk,u=uk))

    for c in stage._boundary_constraints_expr():
      opti.subject_to(stage._expr_apply(c))
        

  def add_objective(self,stage,opti):
    opti.minimize(opti.f+stage._expr_apply(stage.objective))

