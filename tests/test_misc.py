import unittest

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, DirectCollocation, SingleShooting, UniformGrid, GeometricGrid, FreeGrid
from problems import integrator_control_problem, vdp, vdp_dae
from casadi import DM, jacobian, sum1, sum2, MX, rootfinder
from numpy import sin, pi, linspace
from numpy.testing import assert_array_almost_equal
try:
  from contextlib import redirect_stdout
except:
  redirect_stdout = None
from io import StringIO
import numpy as np



class MiscTests(unittest.TestCase):

    def test_grid_convergence(self):
        f_exact = 4.414113817112848

        DM.set_precision(16)
        fss = []
        for method,order in [
          (lambda M: MultipleShooting(N=6,intg='collocation',intg_options={"number_of_finite_elements": M, "interpolation_order": 4}),6),
          (lambda M: MultipleShooting(N=6,intg='rk',M=M), 4),
          (lambda M: MultipleShooting(N=6,intg='cvodes',intg_options={"reltol":10**(-M),"abstol":10**(-M)}),2),
          (lambda M: DirectCollocation(N=6,M=M),7)]:
          fs = []
          for M in [1,2,4,8]:

            ocp, x1, x2, u = vdp(method(M))
            ocp.solver("ipopt",{"ipopt.tol":1e-14})
            sol = ocp.solve()


            f = sol.value(ocp.objective)
            fs.append(f_exact-f)

          print(fs)
          # An increase of factor 2 in M results in a factor 2**order in objective accuracy
          order_meas = -np.log(np.abs(np.array(fs[1:])/np.array(fs[:-1])))/np.log(2)
          print(order_meas)
          self.assertTrue(np.max(order_meas)>order-0.01)

    def test_spy(self):
      ocp, _, _ = integrator_control_problem()
      ocp.solve()
      ocp.spy()
      import matplotlib.pylab as plt
      self.assertEqual(plt.gca().title._text, "Lagrange Hessian: 101x101,0nz")

    def test_ocp_objective(self):
      ocp, x, u = integrator_control_problem()
      sol = ocp.solve()
      self.assertAlmostEqual(sol.value(ocp.objective),sol.value(ocp.at_tf(x)))

    def test_basic(self):
        for T in [1, 3.2]:
            for M in [1, 2]:
                for u_max in [1, 2]:
                    for t0 in [0, 1]:
                        for x0 in [0, 1]:
                            for method in [MultipleShooting(N=4,M=M,intg='rk'), MultipleShooting(N=4,M=M,intg='cvodes'), MultipleShooting(N=4,M=M,intg='idas'), DirectCollocation(N=4,M=M), SingleShooting(N=4,M=M,intg='rk')]:
                                ocp, x, u = integrator_control_problem(
                                    T, u_max, x0, method, t0
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
                for b in [1.0, 2.0]:
                    for method in [MultipleShooting(N=4, intg='rk'), MultipleShooting(N=4, intg='cvodes'), MultipleShooting(N=4, intg='idas'), DirectCollocation(N=4)]:
                        for pos in ["pre","post"]:
                          if pos=="pre":
                            ocp = Ocp(t0=t0, T=FreeTime(1))
                          else:
                            ocp = Ocp(t0=t0)

                          x = ocp.state()
                          u = ocp.control()

                          ocp.set_der(x, u)

                          ocp.subject_to(u <= b)
                          ocp.subject_to(-b <= u)

                          ocp.add_objective(ocp.T)
                          ocp.subject_to(ocp.at_t0(x) == x0)
                          ocp.subject_to(ocp.at_tf(x) == xf)

                          ocp.solver('ipopt')

                          ocp.method(method)

                          if pos=="post":
                            ocp.set_T(ocp.variable())
                            ocp.set_initial(ocp.T, 1)
                            ocp.subject_to(ocp.T>=0)

                          sol = ocp.solve()

                          self.assertAlmostEqual(sol.value(ocp.t0), t0)
                          self.assertAlmostEqual(sol.value(ocp.T),(xf-x0)/b)
                          self.assertAlmostEqual(sol.value(ocp.tf), t0 + (xf - x0) / b)

                          # issue #91
                          self.assertAlmostEqual(sol.value(ocp.at_t0(x)), x0)
                          self.assertAlmostEqual(sol.value(ocp.at_tf(x)), xf)

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
                    for method in [MultipleShooting(N=4, intg='rk'), MultipleShooting(N=4, intg='cvodes'), MultipleShooting(N=4, intg='idas'), DirectCollocation(N=4)]:
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

                        ocp.method(method)

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
      for stage_method in [MultipleShooting(), DirectCollocation()]:
        ocp, x, u = integrator_control_problem(x0=None,stage_method=stage_method)
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
        if redirect_stdout is not None:
          with StringIO() as buf, redirect_stdout(buf):
            ocp.show_infeasibilities(1e-4)
            out = buf.getvalue()
          self.assertIn("ocp.subject_to(ocp.at_t0(x)==2)",out)
      

    def test_time_dep_ode(self):
        t0 = 1.2
        T = 5.7
        ocp = Ocp(t0=t0,T=5.7)
        
        x = ocp.state()
        ocp.set_der(x, ocp.t**2)

        ocp.subject_to(ocp.at_t0(x)==0)
        
        tf = t0+T
        x_ref = tf**3/3-t0**3/3

        ocp.solver('ipopt')
        opts = {"abstol": 1e-9, "reltol": 1e-9}
        for method in [
                MultipleShooting(intg='rk'),
                MultipleShooting(intg='cvodes',intg_options=opts),
                MultipleShooting(intg='idas',intg_options=opts),
                DirectCollocation()]:
            ocp.method(method)
            sol = ocp.solve()
            ts, xs = sol.sample(x,grid='control')
            x_ref = ts**3/3-t0**3/3
            assert_array_almost_equal(xs,x_ref)


    def test_variables(self):
        N = 10
        ocp = Ocp(t0=2*pi,T=10)
        p = ocp.parameter(grid='control')
        v = ocp.variable(grid='control')
        x = ocp.state()
        ocp.set_der(x, 0)
        ocp.subject_to(ocp.at_t0(x)==0)

        ts = linspace(0, 10, N)

        ocp.add_objective(ocp.integral(sin(v-p)**2,grid='control'))
        ocp.method(MultipleShooting(N=N))
        ocp.solver('ipopt')
        ocp.set_value(p, ts)
        ocp.set_initial(v, ts)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')

        assert_array_almost_equal(xs[:-1], ts)
        ocp.set_initial(v, 0.1+2*pi+ts)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')
        assert_array_almost_equal(xs[:-1], 2*pi+ts)
        ocp.set_initial(v, 0.1+ocp.t)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')
        assert_array_almost_equal(xs[:-1], 2*pi+ts)
        ocp.set_initial(v, 0.1+2*pi)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')
        with self.assertRaises(AssertionError):
          assert_array_almost_equal(xs[:-1], 2*pi+ts)
        #ocp.set_value(p, ocp.t)
        #sol = ocp.solve()
        #_, xs = sol.sample(v, grid='control')
        #assert_array_almost_equal(xs[:-1], linspace(2*pi, 2*pi+1, N))


    def test_integral(self):
        t0 = 1.2
        T = 5.7
        ocp = Ocp(t0=t0,T=T)
        
        x = ocp.state()
        u = ocp.control()
        ocp.set_der(x, u)

        ocp.subject_to(ocp.at_t0(x)==0)
        ocp.subject_to(u<=1)
        f = ocp.integral(x*ocp.t)
        ocp.add_objective(-f) # (t-t0)*t -> t^3/3-t^2/2*t0
        ocp.solver('ipopt')
        opts = {"abstol": 1e-8, "reltol": 1e-8, "quad_err_con": True}
        for method in [
                MultipleShooting(intg='rk'),
                MultipleShooting(intg='cvodes',intg_options=opts),
                #MultipleShooting(intg='idas',intg_options=opts),
                DirectCollocation()]:
            ocp.method(method)
            sol = ocp.solve()
            ts, xs = sol.sample(f,grid='control')
            I = lambda t: t**3/3-t**2/2*t0
            x_ref = I(t0+T)-I(t0)

            assert_array_almost_equal(xs[-1],x_ref)

    def test_collocation_equivalence(self):

      for problem in [vdp, vdp_dae]:

        ocp, x1, x2, u = problem(MultipleShooting(N=6,intg='collocation',intg_options={"number_of_finite_elements": 1, "interpolation_order": 4}))
        ocp.solver("ipopt",{"ipopt.tol":1e-12})
        sol = ocp.solve()

        x1_a = sol.sample(x1, grid='control')[1]
        x2_a = sol.sample(x2, grid='control')[1]
        u_a = sol.sample(u, grid='control')[1]

        ocp, x1, x2, u = problem(DirectCollocation(N=6))
        ocp.solver("ipopt",{"ipopt.tol":1e-12})
        sol = ocp.solve()

        x1_b = sol.sample(x1, grid='control')[1]
        x2_b = sol.sample(x2, grid='control')[1]
        u_b = sol.sample(u, grid='control')[1]

        assert_array_almost_equal(x1_a,x1_b,decimal=12)
        assert_array_almost_equal(x2_a,x2_b,decimal=12)
        assert_array_almost_equal(u_a,u_b,decimal=12)


    def test_dae_methods(self):
     
        ocp, x1, x2, u = vdp_dae(DirectCollocation(N=6,M=1))
        ocp.solver("ipopt")
        sol = ocp.solve()

        uref = sol.sample(u, grid='control')[1]

        zsol = sol.sample(ocp.z, grid='control')[1]
        zref = sol.sample(1 - x2**2, grid='control')[1]
        assert_array_almost_equal(zsol,zref,decimal=2)

        for method in [SingleShooting(N=6,intg='idas'),
                       MultipleShooting(N=6,intg='idas'),
                       MultipleShooting(N=6,intg='collocation')]:
           ocp, x1, x2, u = vdp_dae(method)
           ocp.solver("ipopt")
           sol = ocp.solve()

           usol = sol.sample(u, grid='control')[1]
           zsol = sol.sample(ocp.z, grid='control')[1]
           assert_array_almost_equal(zsol[1:],zref[1:],decimal=2)
           assert_array_almost_equal(uref,usol,decimal=3)

        # Constraints with algebraic variables
        ocp, x1, x2, u = vdp_dae(DirectCollocation(N=6,M=1))

        ocp.subject_to(ocp.at_tf(ocp.z) <= 0.8)
        ocp.solver("ipopt")
        sol = ocp.solve()

        zref = sol.sample(ocp.z, grid='control')[1]
        assert_array_almost_equal(zref[-1],0.8,decimal=6)
        uref = sol.sample(u, grid='control')[1]

        for method in [#SingleShooting(N=6,intg='idas'),
                       MultipleShooting(N=6,intg='idas'),
                       MultipleShooting(N=6,intg='collocation')]:
           ocp, x1, x2, u = vdp_dae(method)

           ocp.subject_to(ocp.at_tf(ocp.z) <= 0.8)
           ocp.solver("ipopt")
           sol = ocp.solve()

           usol = sol.sample(u, grid='control')[1]
           zsol = sol.sample(ocp.z, grid='control')[1]
           assert_array_almost_equal(zsol[1:],zref[1:],decimal=2)
           assert_array_almost_equal(uref,usol,decimal=3)

        
        # Path constraints with algebraic variables
        ocp, x1, x2, u = vdp_dae(DirectCollocation(N=6,M=1),x1limit=False)

        ocp.subject_to(0 <= (ocp.z <= 0.8))
        ocp.solver("ipopt")
        sol = ocp.solve()
        ts,zref = sol.sample(ocp.z, grid='control')
        print(zref,ts/10+0.1)
        uref = sol.sample(u, grid='control')[1]

        for method in [SingleShooting(N=6,intg='idas'),
                       MultipleShooting(N=6,intg='idas')]:
                       #MultipleShooting(N=6,intg='collocation')]:
           ocp, x1, x2, u = vdp_dae(method,x1limit=False)

           ocp.subject_to(0 <= (ocp.z <= 0.8))
           ocp.solver("ipopt",{"ipopt.tol": 1e-5})
           sol = ocp.solve()

           usol = sol.sample(u, grid='control')[1]
           zsol = sol.sample(ocp.z, grid='control')[1]
           #assert_array_almost_equal(zsol[1:],zref[1:],decimal=2)
           #assert_array_almost_equal(uref,usol,decimal=3)

           assert_array_almost_equal(np.max(zsol),0.8,decimal=3)
     
    def test_grid_inf_subject_to(self):
        ocp, x1, x2, u = vdp(MultipleShooting(N=10))
        sol = ocp.solve()
        x1sol = sol.sample(x1, grid='integrator',refine=100)[1]
        self.assertFalse(np.all(x1sol>-0.25))

        for method in  [MultipleShooting, DirectCollocation]:
            margins = [np.inf]
            for M in [1,2,4]:
                ocp, x1, x2, u = vdp(method(N=10,M=M),grid='inf')
                sol = ocp.solve()
                x1sol = sol.sample(x1, grid='integrator',refine=100)[1]
                margin = np.min(x1sol-(-0.25))
                self.assertTrue(np.all(margin>0))
                self.assertTrue(np.all(margin<0.01))

                # Assert that margin shrinks for increasing M
                self.assertTrue(margin<0.5*margins[-1])
                margins.append(margin)
                self.assertTrue(margin)

    def test_grid_integrator_subject_to(self):
        ocp, x1, x2, u = vdp(MultipleShooting(N=10))
        sol = ocp.solve()
        x1sol = sol.sample(x1, grid='integrator',refine=100)[1]
        self.assertFalse(np.all(x1sol>-0.25))

        for method, grids in  [(MultipleShooting,['integrator']), (DirectCollocation,['integrator','integrator_roots'])]:
            for grid in grids:
              margins = [np.inf]
              for M in [1,2,4]:
                  ocp, x1, x2, u = vdp(method(N=10,M=M),grid=grid)
                  sol = ocp.solve()
                  x1sol = sol.sample(x1, grid=grid)[1]
                  x1solf = sol.sample(x1, grid='integrator',refine=100)[1]
                  margin = np.min(x1sol-(-0.25))
                  marginf = np.min(x1solf-(-0.25))
                  self.assertTrue(marginf<1e-5)
                  assert_array_almost_equal(margin, 0, decimal=8)

    def test_dae_casadi(self):
        # cross check with dae_colloation

        xref = 0.1 # chariot reference

        l = 1. #- -> crane, + -> pendulum
        m = 1.
        M = 1.
        g = 9.81

        ocp = Ocp(T=5)

        x = ocp.state()
        y = ocp.state()
        w = ocp.state()
        dx = ocp.state()
        dy = ocp.state()
        dw = ocp.state()

        xa = ocp.algebraic()
        u = ocp.control()


        ocp.set_der(x, dx)
        ocp.set_der(y, dy)
        ocp.set_der(w, dw)
        ddx = (w-x)*xa/m
        ddy = g-y*xa/m
        ddw = ((x-w)*xa - u)/M
        ocp.set_der(dx, ddx)
        ocp.set_der(dy, ddy)
        ocp.set_der(dw, ddw)
        ocp.add_alg((x-w)*(ddx - ddw) + y*ddy + dy*dy + (dx-dw)**2)

        ocp.add_objective(ocp.at_tf((x-xref)*(x-xref) + (w-xref)*(w-xref) + dx*dx + dy*dy))
        ocp.add_objective(ocp.integral((x-xref)*(x-xref) + (w-xref)*(w-xref)))

        ocp.subject_to(-2 <= (u <= 2))

        ocp.subject_to(ocp.at_t0(x)==0)
        ocp.subject_to(ocp.at_t0(y)==l)
        ocp.subject_to(ocp.at_t0(w)==0)
        ocp.subject_to(ocp.at_t0(dx)==0)
        ocp.subject_to(ocp.at_t0(dy)==0)
        ocp.subject_to(ocp.at_t0(dw)==0)
        ocp.subject_to(xa>=0,grid='integrator_roots')

        ocp.set_initial(y, l)
        ocp.set_initial(xa, 9.81)

        # Pick an NLP solver backend
        # NOTE: default scaling strategy of MUMPS leads to a singular matrix error 
        ocp.solver('ipopt',{"ipopt.linear_solver": "mumps","ipopt.mumps_scaling":0,"ipopt.tol":1e-12} )

        # Pick a solution method
        method = DirectCollocation(N=50)
        ocp.method(method)

        # Solve
        sol = ocp.solve()

        assert_array_almost_equal(sol.sample(xa, grid='integrator',refine=1)[1][0], 9.81011622448889)
        assert_array_almost_equal(sol.sample(xa, grid='integrator',refine=1)[1][1], 9.865726317147214)
        assert_array_almost_equal(sol.sample(xa, grid='integrator')[1][0], 9.81011622448889)
        assert_array_almost_equal(sol.sample(xa, grid='integrator')[1][1], 9.865726317147214)
        assert_array_almost_equal(sol.sample(xa, grid='control')[1][0], 9.81011622448889)
        assert_array_almost_equal(sol.sample(xa, grid='control')[1][1], 9.865726317147214)

    def test_to_function(self):
        ocp = Ocp(T=FreeTime(1))

        p = ocp.state()
        v = ocp.state()
        u = ocp.control()
        u2 = ocp.control()

        ocp.set_der(p, v)
        ocp.set_der(v, u+u2)

        pp = ocp.parameter()
        ocp.subject_to(u <= p)
        ocp.subject_to(-1 <= u)
        ocp.subject_to(u2 <= p)
        ocp.subject_to(-1 <= u2)

        ocp.add_objective(ocp.T)
        ocp.subject_to(ocp.at_t0(p) == 0)
        ocp.subject_to(ocp.at_t0(v) == 0)
        ocp.subject_to(ocp.at_tf(p) == 1)
        ocp.subject_to(ocp.at_tf(v) == 0)

        ocp.solver('ipopt')

        N = 6
        ocp.method(MultipleShooting(N=N))

        ocp.set_value(pp, 1)
        sol = ocp.solve()

        f = ocp.to_function('f',[ocp.value(pp,grid='control'),ocp.sample(p,grid='control')[1], ocp.sample(v,grid='control')[1], ocp.sample(u,grid='control-')[1]],[ocp.sample(p,grid='control')[1]])

        print(f(1,1,2,3))

    def test_no_solver(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      p = ocp.parameter()
      ocp.set_der(x, u*p)
      with self.assertRaisesRegex(Exception, "You forgot to declare a solver"):
        ocp.solve()

    def test_no_method(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      p = ocp.parameter()
      ocp.set_der(x, u*p)
      ocp.solver('ipopt')
      with self.assertRaisesRegex(Exception, "You forgot to declare a method"):
        ocp.solve()

    def test_no_objective(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      ocp.set_der(x, u)
      ocp.solver('ipopt')
      ocp.method(MultipleShooting(N=3))
      ocp.solve()

    def test_no_solve(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      ocp.set_der(x, u)
      ocp.solver('ipopt')
      ocp.method(MultipleShooting(N=3))
      with self.assertRaisesRegex(Exception, "You forgot to solve first"):
        ocp.non_converged_solution

    def test_no_set_value(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      p = ocp.parameter()
      ocp.set_der(x, u*p)
      ocp.solver('ipopt')
      ocp.method(MultipleShooting(N=3))
      with self.assertRaisesRegex(Exception, "You forgot to declare a value"):
        ocp.solve()

    def test_set_value_error(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      p = ocp.parameter()
      ocp.set_der(x, u*p)
      ocp.solver('ipopt')
      ocp.method(MultipleShooting(N=3))

      y = MX.sym('y')

      ocp.set_value(p, 2)
      with self.assertRaisesRegex(Exception, "You attempted to set the value of a non-parameter"):
        ocp.set_value(y, 3)
      with self.assertRaisesRegex(Exception, "You attempted to set the value of a non-parameter"):
        ocp.set_value(x, 3)


      ocp.solve()
      ocp.set_value(p, 2)
      with self.assertRaisesRegex(Exception, "You attempted to set the value of a non-parameter"):
        ocp.set_value(x, 3)
      with self.assertRaisesRegex(Exception, "You attempted to set the value of a non-parameter"):
        ocp.set_value(y, 3)

    def test_set_initial(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      p = ocp.parameter()
      y = MX.sym('y')
      ocp.set_der(x, u*p)
      ocp.solver('ipopt')
      ocp.method(MultipleShooting(N=3))
      ocp.set_value(p, 2)

      ocp.set_initial(x, 2)
      with self.assertRaisesRegex(Exception, "You attempted to set the initial value of a parameter"):
        ocp.set_initial(p, 3)
      with self.assertRaisesRegex(Exception, "You attempted to set the initial value of an unknown symbol"):
        ocp.set_initial(y, 3)
      ocp.solve()
      ocp.set_initial(x, 2)
      with self.assertRaisesRegex(Exception, "You attempted to set the initial value of a parameter"):
        ocp.set_initial(p, 3)
      with self.assertRaisesRegex(Exception, "You attempted to set the initial value of an unknown symbol"):
        ocp.set_initial(y, 3)

    def test_control_set_der(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      with self.assertRaisesRegex(Exception, "You used set_der on a non"):
        ocp.set_der(u, 2)

    def test_localize_time(self):
      N = 10
      for t0_stage in [FreeTime(-1), -1]:
          for T_stage in [FreeTime(2), 2]:
              t0_free = isinstance(t0_stage, FreeTime)
              T_free = isinstance(T_stage, FreeTime)
              ocp = Ocp(t0=t0_stage,T=T_stage)

              p = ocp.state()
              v = ocp.state()
              u = ocp.control()

              ocp.set_der(p, v)
              ocp.set_der(v, u)

              ocp.add_objective(ocp.tf)
              ocp.subject_to(ocp.at_t0(p) == 0)
              ocp.subject_to(ocp.at_t0(v) == 0)
              ocp.subject_to(ocp.at_tf(p) == 1)
              ocp.subject_to(ocp.at_tf(v) == 0)

              ocp.subject_to(u <= 1)
              ocp.subject_to(-1 <= u)
 
              ocp.solver('ipopt',{"ipopt.tol":1e-12})

              if t0_free:
                  ocp.subject_to(ocp.t0 >= -1)

              for localize_t0 in [False,True]:
                if (not t0_free) and localize_t0: continue
                for localize_T in [False,True]:
                  if (not T_free) and localize_T: continue
  
                  ocp.method(MultipleShooting(N=N,grid=UniformGrid(localize_T=localize_T,localize_t0=localize_t0)))

                  sol = ocp.solve()
                  opti = ocp._method.opti
                  self.assertTrue(sum2(sum1(DM(jacobian(opti.g, opti.x).sparsity(),1))>N)<= t0_free+T_free-localize_t0-localize_T)
                    

    def test_control_grid(self):
      N = 4
      # Be careful for loss of symmetry for non-uniform grids; bang-bang is only optimal if normalized(t=0.5) is included
      # e.g. for local growth factor r: (1-r^7)/(1-r^10) = 0.5 -> r = 
      r = MX.sym("r")
      rf = rootfinder("rf","newton",{"x": r, "g": (1-r**3)/(1-r**N)-0.5})
      r = float(rf(x0=1.2)["x"])
      for t0_stage in [FreeTime(-1), -1]:
          for T_stage in [FreeTime(2), 2]:
              t0_free = isinstance(t0_stage, FreeTime)
              T_free = isinstance(T_stage, FreeTime)
              ocp = Ocp(t0=t0_stage,T=T_stage)

              p = ocp.state()
              v = ocp.state()
              u = ocp.control()

              ocp.set_der(p, v)
              ocp.set_der(v, u)

              ocp.add_objective(ocp.tf)
              ocp.subject_to(ocp.at_t0(p) == 0)
              ocp.subject_to(ocp.at_t0(v) == 0)
              ocp.subject_to(ocp.at_tf(p) == 1)
              ocp.subject_to(ocp.at_tf(v) == 0)


              ocp.subject_to(-1 <= (u <= 1))
              ocp.solver('ipopt',{"ipopt.tol":1e-12})

              if t0_free:
                  ocp.subject_to(ocp.t0 >= -1)

              for localize_t0 in [False,True]:
                if (not t0_free) and localize_t0: continue
                for localize_T in [False,True]:
                  if (not T_free) and localize_T: continue

                  for grid in [
                    UniformGrid(localize_T=localize_T,localize_t0=localize_t0),
                    GeometricGrid(r,local=True,localize_T=localize_T,localize_t0=localize_t0),
                    GeometricGrid(r**(N-1),local=False,localize_T=localize_T,localize_t0=localize_t0),
                    FreeGrid(min=0.01,max=0.5)]:
                      ocp.method(MultipleShooting(N=N,grid=grid))

                      sol = ocp.solve()

                      tolerance = 1e-6

                      if isinstance(grid, UniformGrid):

                        ts_ref = -1 + np.linspace(0, 2, 4*N+1)

                        ts, _ = sol.sample(v, grid='integrator', refine=4)
                        np.testing.assert_allclose(ts, ts_ref, atol=tolerance)

                      ts, _ = sol.sample(v, grid='control')
                      if not isinstance(grid, FreeGrid):

                        ts_ref = -1 + 2*np.array(grid.normalized(N))

                        np.testing.assert_allclose(ts, ts_ref, atol=tolerance)

                      np.testing.assert_allclose(ts[0], -1, atol=tolerance)
                      np.testing.assert_allclose(ts[-1], 1, atol=tolerance)

                      self.assertAlmostEqual(sol.value(ocp.t0),-1, places=5)
                      self.assertAlmostEqual(sol.value(ocp.T),2, places=5)
                      self.assertAlmostEqual(sol.value(ocp.tf),1, places=5)

if __name__ == '__main__':
    unittest.main()
