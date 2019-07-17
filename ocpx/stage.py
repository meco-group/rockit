from collections import OrderedDict
from casadi import MX, substitute, Function, vcat, depends_on, vertcat

class Stage:
  def __init__(self, ocp, t0=0, tf=1):
    self.ocp = ocp
    self.states = OrderedDict()
    self.controls = OrderedDict()
    self._state_der = dict()
    self._constraints = []
    self._expr_t0 = OrderedDict() # Expressions defined at t0
    self._expr_tf = OrderedDict() # Expressions defined at tf
    self._objective = 0
    self.t0 = t0
    self.tf = tf

  def state(self):
    """
    Create a state
    """
    x = MX.sym("x")
    self.states[x] = x
    return x

  def control(self):
    u = MX.sym("u")
    self.controls[u] = u
    return u

  def set_der(self, state, der):
    self._state_der[state] = der

  def integral(self,expr):
    I = self.state()
    self.set_der(I, expr)
    self.subject_to(self.at_t0(I)==0)
    return self.at_tf(I)

  def subject_to(self, constr):
    self._constraints.append(constr)

  def at_t0(self, expr):
    p = MX.sym("p_t0", expr.sparsity())
    self._expr_t0[p] = expr
    return p

  def at_tf(self, expr):
    p = MX.sym("p_tf", expr.sparsity())
    self._expr_tf[p] = expr
    return p

  def add_objective(self, term):
    self._objective = self._objective + term

  def method(self,method):
    self._method = method

  @property
  def x(self):
    return vcat(list(self.states.values()))

  @property
  def u(self):
    return vcat(list(self.controls.values()))

  @property
  def nx(self):
    return self.x.numel()

  @property
  def nu(self):
    return self.u.numel()

  # Internal methods

  def _ode_dict(self):
    ode = {}
    ode['x'] = self.x
    ode['p'] = self.u
    ode['ode'] = vcat([self._state_der[k] for k in self.states.keys()])
    return ode

  def _bake(self,x0=None,xf=None,u0=None,uf=None):
    for k in self._expr_t0.keys():
      self._expr_t0[k] = substitute([self._expr_t0[k]],[self.x,self.u],[x0,u0])[0]
    for k in self._expr_tf.keys():
      self._expr_tf[k] = substitute([self._expr_tf[k]],[self.x,self.u],[xf,uf])[0]

  def _boundary_constraints_expr(self):
    return [c for c in self._constraints if not self.is_trajectory(c)]

  def _path_constraints_expr(self):
    return [c for c in self._constraints if self.is_trajectory(c)]

  def _expr_apply(self,expr,**kwargs):
    expr = substitute([expr],
      list(self._expr_t0.keys())+list(self._expr_tf.keys()),
      list(self._expr_t0.values())+list(self._expr_tf.values()))[0]
    if kwargs:
      helper = self._expr_to_function(expr)
      return helper.call(kwargs,True,False)["out"]
    else:
      return expr
  _constr_apply = _expr_apply

  def _expr_to_function(self,expr):
    return Function('helper',[self.x,self.u],[expr],["x","u"],["out"])

  def is_trajectory(self, expr):
    return depends_on(expr,vertcat(self.x,self.u))
