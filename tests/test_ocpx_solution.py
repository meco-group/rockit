import unittest
import numpy as np
from numpy.testing import assert_allclose

from ocpx import *


class MiscTests(unittest.TestCase):
    def test_grid_integrator(self):
        N, T, u_max, x0 = 10, 10, 2, 1

        ocp = OcpMultiStage()
        stage = ocp.stage(t0=0, T=T)

        x = stage.state()
        u = stage.control()

        stage.set_der(x, u)

        stage.subject_to(u <= u_max)
        stage.subject_to(-u_max <= u)

        stage.add_objective(stage.at_tf(x))
        stage.subject_to(stage.at_t0(x) == x0)

        ocp.method(DirectMethod(solver='ipopt'))

        stage.method(MultipleShooting(N=N, M=3, intg='rk'))

        sol = ocp.solve()

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
        assert_allclose(ts, t_exact, atol=1e-6)
        assert_allclose(xs[1:], x_exact[:-1], atol=1e-6)
        assert_allclose(us, u_exact, atol=1e-6)
        assert_allclose(uxs[1:], u_exact[:-1] * x_exact[:-1], atol=1e-6)
