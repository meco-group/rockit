import unittest
import numpy as np
from numpy.testing import assert_allclose
from problems import integrator_control_problem, bang_bang_problem
from ocpx import MultipleShooting

class OcpSolutionTests(unittest.TestCase):
    def test_grid_integrator(self):
        N, T, u_max, x0 = 10, 10, 2, 1
        tolerance = 1e-6
        ocp, x, u = integrator_control_problem(T, u_max, x0, MultipleShooting(N=N,M=3,intg='rk'))
        sol = ocp.solve()
        ts, xs = sol.sample(x, grid='integrator')
        ts, us = sol.sample(u, grid='integrator')
        ts, uxs = sol.sample(u * x, grid='integrator')

        t_exact = np.linspace(0, T, N * 3 + 1)
        x_exact = np.linspace(1, x0 - 10 * u_max, N * 3 + 1)
        u_exact = np.ones(N * 3 + 1) * (-u_max)

        assert_allclose(ts, t_exact, atol=tolerance)
        assert_allclose(xs, x_exact, atol=tolerance)
        assert_allclose(us, u_exact, atol=tolerance)
        assert_allclose(uxs, u_exact * x_exact, atol=tolerance)

    def test_intg_refine(self):
        ocp, sol, p, v, u = bang_bang_problem(MultipleShooting(N=2,intg='rk'))
        tolerance = 1e-6

        ts, ps = sol.sample(p, grid='integrator', refine=10)

        ps_ref = np.hstack(((0.5*np.linspace(0,1, 10+1)**2)[:-1],np.linspace(0.5,1.5,10+1)-0.5*np.linspace(0,1, 10+1)**2)) 
        assert_allclose(ps, ps_ref, atol=tolerance)

        ts_ref = np.linspace(0, 2, 10*2+1)

        ts, vs = sol.sample(v, grid='integrator', refine=10)
        assert_allclose(ts, ts_ref, atol=tolerance)

        vs_ref = np.hstack((np.linspace(0,1, 10+1)[:-1],np.linspace(1,0, 10+1))) 
        assert_allclose(vs, vs_ref, atol=tolerance)


        u_ref = np.array([1.0]*10+[-1.0]*11)
        ts, us = sol.sample(u, grid='integrator', refine=10)
        assert_allclose(us, u_ref, atol=tolerance)

if __name__ == '__main__':
    unittest.main()
