import unittest
import numpy as np
from numpy.testing import assert_allclose
from problems import integrator_control_problem
from ocpx import MultipleShooting


class OcpxSolutionTests(unittest.TestCase):
    def test_grid_integrator(self):
        N, T, u_max, x0 = 10, 10, 2, 1
        tolerance = 1e-6
        sol, stage, x, u = integrator_control_problem(T, u_max, x0, MultipleShooting(N=N,M=3,intg='rk'))

        ts, xs = sol.sample(
            stage, x, grid='integrator')
        ts, us = sol.sample(
            stage, u, grid='integrator')
        ts, uxs = sol.sample(
            stage, u * x, grid='integrator')

        t_exact = np.linspace(0, T, N * 3 + 1)
        x_exact = np.linspace(1, x0 - 10 * u_max, N * 3 + 1)
        u_exact = np.ones(N * 3 + 1) * (-u_max)

        print(xs)
        print(x_exact)

        assert_allclose(ts, t_exact, atol=tolerance)
        assert_allclose(xs, x_exact, atol=tolerance)
        assert_allclose(us, u_exact, atol=tolerance)
        assert_allclose(uxs, u_exact * x_exact, atol=tolerance)

if __name__ == '__main__':
    unittest.main()
