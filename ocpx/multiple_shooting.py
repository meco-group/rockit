from .sampling_method import SamplingMethod

class MultipleShooting(SamplingMethod):
  def __init__(self,*args,**kwargs):
    SamplingMethod.__init__(self,*args,**kwargs)
    self.X = []
    self.U = []

  def transcribe(self,stage,opti):
    self.add_variables(stage,opti)
    stage._bake(x0=self.X[0],
               xf=self.X[-1],
               u0=self.U[0],
               uf=self.U[-1])
    self.add_constraints(stage,opti)
    self.add_objective(stage,opti)

  def add_variables(self,stage,opti):
    self.X.append(opti.variable(stage.nx))

    for k in range(self.N):
      self.U.append(opti.variable(stage.nu))
      self.X.append(opti.variable(stage.nx))

  def add_constraints(self,stage,opti):
    F = self.discrete_system(stage)

    for k in range(self.N):
      xk = self.X[k]
      uk = self.U[k]
      xk_next = self.X[k+1]
      opti.subject_to(xk_next==F(x0=xk,p=uk)["xf"])

      for c in stage._path_constraints_expr():
        opti.subject_to(stage._expr_apply(c,x=xk,u=uk))

    for c in stage._boundary_constraints_expr():
      opti.subject_to(stage._expr_apply(c))
        

  def add_objective(self,stage,opti):
    opti.minimize(opti.f+stage._expr_apply(stage._objective))

