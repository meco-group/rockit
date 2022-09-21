import unittest

from rockit import Ocp, DirectMethod, MultipleShooting, DirectCollocation, SingleShooting, SplineMethod
from problems import integrator_control_problem


class MethodTests(unittest.TestCase):

    def test_all(self):
      T = 1
      M = 1
      b = 1
      t0 = 0
      x0 = 0
      for scheme in [MultipleShooting(N=40, M=1, intg='rk'),
                     DirectCollocation(N=40), SingleShooting(N=40, M=1, intg='rk'),
                     SplineMethod(N=10)]:
        (ocp, x, u) = integrator_control_problem(T, b, x0, scheme, t0)
        sol = ocp.solve()
        ts, xs = sol.sample(x, grid='control')

        self.assertAlmostEqual(xs[0],x0,places=6)
        self.assertAlmostEqual(xs[-1],x0-b*T,places=6)
  
if __name__ == '__main__':
    unittest.main()
