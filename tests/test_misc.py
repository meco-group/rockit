import unittest

from ocpx import *

class MiscTests(unittest.TestCase):

    def test_basic(self):
      ocp = OcpMultiStage()
      stage = ocp.stage(t0=0,T=1)

      x = stage.state()
      u = stage.control()

      stage.set_der(x,u)

      stage.subject_to(u<=1)
      stage.subject_to(-1<=u)
      
      stage.add_objective(stage.at_tf(x))
      stage.subject_to(stage.at_t0(x)==0)

      ocp.method(DirectMethod(solver='ipopt'))

      stage.method(MultipleShooting(N=4,intg='rk'))
      
      sol = ocp.solve()

      ts, xs = sol.sample(stage,x,grid=stage.grid_control)

      self.assertAlmostEqual(xs[-1],-1)

if __name__ == '__main__':
    unittest.main()
