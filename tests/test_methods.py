import unittest

from pylab import *

from rockit import Ocp, DirectMethod, MultipleShooting, DirectCollocation, SingleShooting, SplineMethod
from problems import integrator_control_problem, bang_bang_chain_problem
import numpy as np

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
  

    def test_spline_more_vars(self):
      (ocp, p, v, a) = bang_bang_chain_problem(MultipleShooting(N=10, intg='rk'))
      sol = ocp.solve()
      ts, ref_ps = sol.sample(p, grid='control')
      ts, ref_vs = sol.sample(v, grid='control')
      ts, ref_as = sol.sample(a, grid='control')
      for scheme in [DirectCollocation(N=10), SingleShooting(N=10, intg='rk'), SplineMethod(N=10)]:
        (ocp, p, v, a) = bang_bang_chain_problem(scheme)
        sol = ocp.solve()
        ts, ps = sol.sample(p, grid='control')
        ts, vs = sol.sample(v, grid='control')
        ts, ass = sol.sample(a, grid='control')

        np.testing.assert_allclose(ps, ref_ps, atol=1e-6)
        np.testing.assert_allclose(vs, ref_vs, atol=1e-6)
        np.testing.assert_allclose(ass, ref_as, atol=1e-6)
        

        
if __name__ == '__main__':
    unittest.main()
