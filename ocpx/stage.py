from collections import OrderedDict
from casadi import *

class Stage:
  def __init__(self, ocp, t0=0, tf=1):
    self.ocp = ocp
    self.states = OrderedDict()
    self.controls = OrderedDict()
    self.state_der = dict()
    self.constraints = []
    self.expr_t0 = OrderedDict() # Expressions defined at t0
    self.expr_tf = OrderedDict() # Expressions defined at tf
    self.objective = 0
    self.t0 = t0
    self.tf = tf

  def state(self):
    x = MX.sym("x")
    self.states[x] = x
    return x

  def control(self):
    u = MX.sym("u")
    self.controls[u] = u
    return u

  def set_der(self, state, der):
    self.state_der[state] = der

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

  def ode_dict(self):
    ode = {}
    ode['x'] = self.x
    ode['p'] = self.u
    ode['ode'] = vcat([self.state_der[k] for k in self.states.keys()])
    return ode

  def integral(self,expr):
    I = self.state()
    self.set_der(I, expr)
    self.subject_to(self.at_t0(I)==0)
    return self.at_tf(I)

  def subject_to(self, constr):
    self.constraints.append(constr)

  def at_t0(self, expr):
    p = MX.sym("p_t0", expr.sparsity())
    self.expr_t0[p] = expr
    return p

  def at_tf(self, expr):
    p = MX.sym("p_tf", expr.sparsity())
    self.expr_tf[p] = expr
    return p

  def _bake(self,x0=None,xf=None,u0=None,uf=None):
    for k in self.expr_t0.keys():
      self.expr_t0[k] = substitute([self.expr_t0[k]],[self.x,self.u],[x0,u0])[0]
    for k in self.expr_tf.keys():
      self.expr_tf[k] = substitute([self.expr_tf[k]],[self.x,self.u],[xf,uf])[0]

  def add_objective(self, term):
    self.objective = self.objective + term

  def method(self,method):
    self._method = method

  def _boundary_constraints_expr(self):
    return [c for c in self.constraints if not self.is_trajectory(c)]

  def _path_constraints_expr(self):
    return [c for c in self.constraints if self.is_trajectory(c)]

  def _expr_apply(self,expr,**kwargs):
    expr = substitute([expr],
      list(self.expr_t0.keys())+list(self.expr_tf.keys()),
      list(self.expr_t0.values())+list(self.expr_tf.values()))[0]
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
