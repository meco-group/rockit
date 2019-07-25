import unittest

from ocpx import OcpMultiStage, DirectMethod, MultipleShooting
from problems import integrator_control_problem


class MiscTests(unittest.TestCase):

    def test_basic(self):
        for T in [1, 3.2]:
            for M in [1, 2]:
                for u_max in [1, 2]:
                    for t0 in [0, 1]:
                        for x0 in [0, 1]:
                            for intg_method in ['rk', 'cvodes', 'idas']:
                                sol, stage, x, u = integrator_control_problem(
                                    4, M, T, u_max, x0, t0, intg_method
                                )

                                ts, xs = sol.sample(
                                    stage, x, grid=stage.grid_control)

                                self.assertAlmostEqual(xs[0], x0, places=6)
                                self.assertAlmostEqual(
                                    xs[-1], x0 - u_max * T, places=6)
                                self.assertAlmostEqual(ts[0], t0)
                                self.assertAlmostEqual(ts[-1], t0 + T)

    def test_basic_time_free(self):
        xf = 2
        for t0 in [0, 1]:
            for x0 in [0, 1]:
                for b in [1, 2]:
                    for intg_method in ['rk', 'cvodes', 'idas']:
                        ocp = OcpMultiStage()
                        stage = ocp.stage(t0=t0, T=ocp.free(1))

                        x = stage.state()
                        u = stage.control()

                        stage.set_der(x, u)

                        stage.subject_to(u <= b)
                        stage.subject_to(-b <= u)

                        stage.add_objective(stage.T)
                        stage.subject_to(stage.at_t0(x) == x0)
                        stage.subject_to(stage.at_tf(x) == xf)

                        ocp.method(DirectMethod(solver='ipopt'))

                        stage.method(MultipleShooting(N=4, intg=intg_method))

                        sol = ocp.solve()

                        ts, xs = sol.sample(stage, x, grid=stage.grid_control)

                        self.assertAlmostEqual(xs[0], x0, places=6)
                        self.assertAlmostEqual(xs[-1], xf, places=6)
                        self.assertAlmostEqual(ts[0], t0)
                        self.assertAlmostEqual(ts[-1], t0 + (xf - x0) / b)


if __name__ == '__main__':
    unittest.main()
