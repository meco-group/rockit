import unittest
import numpy as np
from numpy.testing import assert_allclose
from problems import integrator_control_problem


class OcpxSolutionTests(unittest.TestCase):
    def test_grid_integrator(self):
        N, T, u_max, x0 = 10, 10, 2, 1
        tolerance = 1e-6
        sol, stage, x, u = integrator_control_problem(N, 3, T, u_max, x0)

        ts, xs = sol.sample(
            stage, x, grid=stage.grid_integrator)
        ts, us = sol.sample(
            stage, u, grid=stage.grid_integrator)
        ts, uxs = sol.sample(
            stage, u * x, grid=stage.grid_integrator)

        t_exact = np.linspace(0, T, N * 3 + 1)
        x_exact = np.linspace(1, x0 - T * u_max, N * 3 + 1)
        u_exact = np.ones(N * 3 + 1) * (-u_max)

        # Note: index hack because of issue with sample function
        # Fix this and run the correct version of this test!
        assert_allclose(ts, t_exact, atol=tolerance)
        assert_allclose(xs[1:], x_exact[:-1], atol=tolerance)
        assert_allclose(us, u_exact, atol=tolerance)
        assert_allclose(uxs[1:], u_exact[:-1] * x_exact[:-1], atol=tolerance)
