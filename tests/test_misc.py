import unittest

from ocpx import Ocp, DirectMethod, MultipleShooting, FreeTime, DirectCollocation
from problems import integrator_control_problem
from numpy import sin, pi
from contextlib import redirect_stdout
from io import StringIO

class MiscTests(unittest.TestCase):

    def test_spy(self):
      ocp, _, _ = integrator_control_problem()
      ocp.solve()
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
                                ocp, x, u = integrator_control_problem(
                                    T, u_max, x0, MultipleShooting(N=4,M=M,intg=intg_method), t0
                                )
                                sol = ocp.solve()

                                ts, xs = sol.sample(x, grid='control')

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
        ocp = Ocp(t0=t0,T=T)

        x = ocp.state()
        u = ocp.control()

        ocp.set_der(x,u)

        y = 2*x

        ocp.subject_to(ocp.der(y)<=2*b)
        ocp.subject_to(-2*b<=ocp.der(y))
       
        ocp.add_objective(ocp.at_tf(x))
        ocp.subject_to(ocp.at_t0(x)==x0)

        ocp.solver('ipopt')

        ocp.method(MultipleShooting(N=4,M=M,intg='rk'))
       
        sol = ocp.solve()

        ts, xs = sol.sample(x,grid='control')

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
                        ocp = Ocp(t0=t0, T=FreeTime(1))

                        x = ocp.state()
                        u = ocp.control()

                        ocp.set_der(x, u)

                        ocp.subject_to(u <= b)
                        ocp.subject_to(-b <= u)

                        ocp.add_objective(ocp.T)
                        ocp.subject_to(ocp.at_t0(x) == x0)
                        ocp.subject_to(ocp.at_tf(x) == xf)

                        ocp.solver('ipopt')

                        ocp.method(MultipleShooting(N=4, intg=intg_method))

                        sol = ocp.solve()

                        ts, xs = sol.sample(x, grid='control')

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
                        ocp = Ocp(t0=FreeTime(2),T=T)

                        x = ocp.state()
                        u = ocp.control()

                        ocp.set_der(x, u)
                        ocp.subject_to(u <= b)
                        ocp.subject_to(-b <= u)

                        ocp.add_objective(ocp.tf)
                        ocp.subject_to(ocp.at_t0(x) == x0)
                        ocp.subject_to(ocp.at_tf(x) == xf)
                        ocp.subject_to(ocp.t0 >= 0)

                        ocp.solver('ipopt')

                        ocp.method(MultipleShooting(N=4, intg=intg_method))

                        sol = ocp.solve()

                        ts, xs = sol.sample(x, grid='control')

                        self.assertAlmostEqual(xs[0], x0, places=6)
                        self.assertAlmostEqual(xs[-1], xf, places=6)
                        self.assertAlmostEqual(ts[0], t0)
                        self.assertAlmostEqual(ts[-1], t0 + T)

    def test_param(self):
      ocp = Ocp(T=1)

      x = ocp.state()
      u = ocp.control()

      p = ocp.parameter()

      ocp.set_der(x, u)

      ocp.subject_to(u <= 1)
      ocp.subject_to(-1 <= u)

      ocp.add_objective(ocp.at_tf(x))
      ocp.subject_to(ocp.at_t0(x) == p)

      ocp.solver('ipopt')

      ocp.method(MultipleShooting())

      ocp.set_value(p, 0)
      sol = ocp.solve()

      ts, xs = sol.sample(x, grid='control')
      self.assertAlmostEqual(xs[0], 0)

      ocp.set_value(p, 1)
      sol = ocp.solve()

      ts, xs = sol.sample(x, grid='control')
      self.assertAlmostEqual(xs[0], 1)

    def test_initial(self):
      ocp, x, u = integrator_control_problem(x0=None)
      v = ocp.variable()
      ocp.subject_to(ocp.at_t0(x)==v)
      ocp.subject_to(0==sin(v))
      sol = ocp.solve()
      ts, xs = sol.sample(x, grid='control')
      self.assertAlmostEqual(xs[0], 0, places=6)

      ocp.set_initial(v, 2*pi)
      sol = ocp.solve()
      ts, xs = sol.sample(x, grid='control')
      self.assertAlmostEqual(xs[0], 2*pi, places=6)

    def test_show_infeasibilities(self):
      for method in [MultipleShooting(), DirectCollocation()]:
        ocp, x, u = integrator_control_problem(stage_method=method, x0 = 0)
        ocp.subject_to(ocp.at_t0(x)==2)   
        with self.assertRaises(Exception):
          sol = ocp.solve()
        with StringIO() as buf, redirect_stdout(buf):
          ocp.show_infeasibilities(1e-4)
          out = buf.getvalue()
        self.assertIn("ocp.subject_to(ocp.at_t0(x)==2)",out)
      
if __name__ == '__main__':
    unittest.main()
