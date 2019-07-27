import unittest

from ocpx import OcpMultiStage, DirectMethod, MultipleShooting
from problems import integrator_control_problem


class MiscTests(unittest.TestCase):

    def test_spy(self):
      ocp, _, _, _, _ = integrator_control_problem()
      ocp.spy()
      import matplotlib.pylab as plt
      self.assertEqual(plt.gca().title._text, "Lagrange Hessian: 101x101,0nz")


    def test_basic(self):
        for T in [1, 3.2]:
            for M in [1, 2]:
                for u_max in [1, 2]:
                    for t0 in [0, 1]:
                        for x0 in [0, 1]:
                            for intg_method in ['rk', 'cvodes', 'idas']:
                                _, sol, stage, x, u = integrator_control_problem(
                                    T, u_max, x0, MultipleShooting(N=4,M=M,intg=intg_method), t0
                                )

                                ts, xs = sol.sample(
                                    stage, x, grid='control')

                                self.assertAlmostEqual(xs[0], x0, places=6)
                                self.assertAlmostEqual(
                                    xs[-1], x0 - u_max * T, places=6)
                                self.assertAlmostEqual(ts[0], t0)
                                self.assertAlmostEqual(ts[-1], t0 + T)

    def test_der(self):
        T = 1
        M = 1
        b = 1
        t0 = 0
        x0 = 0
        ocp = OcpMultiStage()
        stage = ocp.stage(t0=t0,T=T)

        x = stage.state()
        u = stage.control()

        stage.set_der(x,u)

        y = 2*x

        stage.subject_to(stage.der(y)<=2*b)
        stage.subject_to(-2*b<=stage.der(y))
       
        stage.add_objective(stage.at_tf(x))
        stage.subject_to(stage.at_t0(x)==x0)

        ocp.method(DirectMethod(solver='ipopt'))

        stage.method(MultipleShooting(N=4,M=M,intg='rk'))
       
        sol = ocp.solve()

        ts, xs = sol.sample(stage,x,grid='control')

        self.assertAlmostEqual(xs[0],x0,places=6)
        self.assertAlmostEqual(xs[-1],x0-b*T,places=6)
        self.assertAlmostEqual(ts[0],t0)
        self.assertAlmostEqual(ts[-1],t0+T)


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

                        ts, xs = sol.sample(stage, x, grid='control')

                        self.assertAlmostEqual(xs[0], x0, places=6)
                        self.assertAlmostEqual(xs[-1], xf, places=6)
                        self.assertAlmostEqual(ts[0], t0)
                        self.assertAlmostEqual(ts[-1], t0 + (xf - x0) / b)

    def test_basic_t0_free(self):
        xf = 2
        t0 = 0
        for T in [2]:
            for x0 in [0, 1]:
                for b in [1, 2]:
                    for intg_method in ['rk', 'cvodes', 'idas']:
                        ocp = OcpMultiStage()
                        stage = ocp.stage(t0=ocp.free(2),T=T)

                        x = stage.state()
                        u = stage.control()

                        stage.set_der(x, u)
                        stage.subject_to(u <= b)
                        stage.subject_to(-b <= u)

                        stage.add_objective(stage.tf)
                        stage.subject_to(stage.at_t0(x) == x0)
                        stage.subject_to(stage.at_tf(x) == xf)
                        stage.subject_to(stage.t0 >= 0)

                        ocp.method(DirectMethod(solver='ipopt'))

                        stage.method(MultipleShooting(N=4, intg=intg_method))

                        sol = ocp.solve()

                        ts, xs = sol.sample(stage, x, grid='control')

                        self.assertAlmostEqual(xs[0], x0, places=6)
                        self.assertAlmostEqual(xs[-1], xf, places=6)
                        self.assertAlmostEqual(ts[0], t0)
                        self.assertAlmostEqual(ts[-1], t0 + T)

    def test_param(self):
      ocp = OcpMultiStage()
      stage = ocp.stage(T=1)

      x = stage.state()
      u = stage.control()

      p = stage.parameter()

      stage.set_der(x, u)

      stage.subject_to(u <= 1)
      stage.subject_to(-1 <= u)

      stage.add_objective(stage.at_tf(x))
      stage.subject_to(stage.at_t0(x) == p)

      ocp.method(DirectMethod(solver='ipopt'))

      stage.method(MultipleShooting())

      stage.set_value(p, 0)
      sol = ocp.solve()

      ts, xs = sol.sample(stage, x, grid='control')
      self.assertAlmostEqual(xs[0], 0)

      stage.set_value(p, 1)
      sol = ocp.solve()

      ts, xs = sol.sample(stage, x, grid='control')
      self.assertAlmostEqual(xs[0], 1)

if __name__ == '__main__':
    unittest.main()
