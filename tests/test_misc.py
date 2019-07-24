import unittest

from ocpx import *

class MiscTests(unittest.TestCase):

    def test_basic(self):

      for T in [1,3.2]:
        for M in [1,2]:
          for b in [1,2]:
            for t0 in [0,1]:
              for x0 in [0,1]:
                ocp = OcpMultiStage()
                stage = ocp.stage(t0=t0,T=T)

                x = stage.state()
                u = stage.control()

                stage.set_der(x,u)

                stage.subject_to(u<=b)
                stage.subject_to(-b<=u)
                
                stage.add_objective(stage.at_tf(x))
                stage.subject_to(stage.at_t0(x)==x0)

                ocp.method(DirectMethod(solver='ipopt'))

                stage.method(MultipleShooting(N=4,M=M,intg='rk'))
                
                sol = ocp.solve()

                ts, xs = sol.sample(stage,x,grid=stage.grid_control)

                self.assertAlmostEqual(xs[0],x0,places=6)
                self.assertAlmostEqual(xs[-1],x0-b*T,places=6)
                self.assertAlmostEqual(ts[0],t0)
                self.assertAlmostEqual(ts[-1],t0+T)


if __name__ == '__main__':
    unittest.main()
