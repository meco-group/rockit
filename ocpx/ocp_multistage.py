from .stage import Stage
from .freetime import FreeTime
from casadi import hcat
from .ocpx_solution import OcpxSolution
from copy import deepcopy

class OcpMultiStage:
  def __init__(self):
    self.stages = []
    # Flag to make solve() faster when solving a second time (e.g. with different parameter values)
    self.is_transcribed = False

  def stage(self, prev_stage=None, **kwargs):
      if prev_stage is None:
          s = Stage(self,**kwargs)
      else:
          s = deepcopy(prev_stage)

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
    return OcpxSolution(opti.solve())

  def free(self,T_init):
    return FreeTime(T_init)
