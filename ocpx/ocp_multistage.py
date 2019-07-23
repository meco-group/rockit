from .stage import Stage
from casadi import hcat

class OcpMultiStage:
  def __init__(self):
    self.stages = []
    # Flag to make solve() faster when solving a second time (e.g. with different parameter values)
    self.is_transcribed = False

  def stage(self,**kwargs):
    s = Stage(self,**kwargs)
    self.stages.append(s)
    return s

  def method(self,method):
    self._method = method

  def solve(self):
    opti = self._method.opti
    if not self.is_transcribed:
      for s in self.stages:
        s._method.transcribe(s, opti)
      self.is_transcribed = True
    sol = opti.solve()

  def free(self,T_init):
    return FreeTime(T_init)

class FreeTime:
  def __init__(self, T_init):
    self.T_init = T_init
