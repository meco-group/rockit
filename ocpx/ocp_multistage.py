from .stage import Stage

class OcpMultiStage:
  def __init__(self):
    self.stages = []

  def stage(self,**kwargs):
    s = Stage(self,**kwargs)
    self.stages.append(s)
    return s

  def method(self,method):
    self._method = method

  def solve(self):
    opti = self._method.opti
    for s in self.stages:
      s._method.transcribe(s, opti)
    sol = opti.solve()