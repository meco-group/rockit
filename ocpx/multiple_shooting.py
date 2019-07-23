from .sampling_method import SamplingMethod
from casadi import sumsqr, vertcat

class MultipleShooting(SamplingMethod):
  def __init__(self,*args,**kwargs):
    SamplingMethod.__init__(self,*args,**kwargs)
    self.X = [] # List that will hold N+1 decision variables for state vector
    self.U = [] # List that will hold N decision variables for control vector

  def transcribe(self,stage,opti):
    """
    Transcription is the process of going from a continous-time OCP to an NLP
    """
    self.add_variables(stage,opti)
    # Now that decision variables exist, we can bake the at_t0(...)/at_tf(...) expressions
    stage._bake(x0=self.X[0],xf=self.X[-1],
                u0=self.U[0],uf=self.U[-1])
    self.add_constraints(stage,opti)
    self.add_objective(stage,opti)

  def add_variables(self,stage,opti):
    # We are creating variables in a special order such that the resulting constraint Jacobian
    # is block-sparse
    self.X.append(opti.variable(stage.nx))
    self.T=opti.variable()

    for k in range(self.N):
      self.U.append(opti.variable(stage.nu))
      self.X.append(opti.variable(stage.nx))

  def add_constraints(self,stage,opti):
    # Obtain the discretised system
    F = self.discrete_system(stage)
    opti.subject_to(self.T>=0)

    for k in range(self.N):
      # Dynamic constraints a.k.a. gap-closing constraints
      opti.subject_to(self.X[k+1]==F(x0=self.X[k],u=self.U[k],T=self.T)["xf"])

      for c in stage._path_constraints_expr(): # for each constraint expression
        # Add it to the optimizer, but first make x,u concrete.
        opti.subject_to(stage._constr_apply(c,x=self.X[k],u=self.U[k],T = self.T))

    for c in stage._boundary_constraints_expr(): # Append boundary conditions to the end
      opti.subject_to(stage._constr_apply(c))
        
  def add_objective(self,stage,opti):
    opti.minimize(opti.f+stage._expr_apply(stage._objective,T=self.T))

