from ast import Mult
import unittest

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, DirectCollocation, SingleShooting, SplineMethod, UniformGrid, GeometricGrid, FreeGrid, LseGroup, rockit_pickle_context, rockit_unpickle_context
from problems import integrator_control_problem, vdp, vdp_dae, bang_bang_problem
from casadi import DM, jacobian, sum1, sum2, MX, rootfinder, evalf, sumsqr, symvar, vertcat
import casadi as ca
from numpy import sin, pi, linspace
from numpy.testing import assert_array_almost_equal
from rockit.splines.spline import Spline
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

    def test_splinemethod_order0(self):
      for method in [MultipleShooting(N=10),SplineMethod(N=10)]:
         for order in [0,1]:
            # Only control
            ocp = Ocp(T=1)

            x = ocp.control(order=order)
            ocp.add_objective(ocp.sum(sumsqr(x-ocp.t),include_last=True))
            ocp.solver('ipopt')

            ocp.method(method)

            # Solve
            sol = ocp.solve()
            [t,xs] = sol.sample(x, grid='control')
            if order==1:
              assert_array_almost_equal(t, xs)
            if order==0:
              assert_array_almost_equal(t[:-2], xs[:-2])
              assert_array_almost_equal(xs[-2:],0.95)

            if "SplineMethod" in str(method):
              print(sol.sample(x, grid='gist'))

            ocp = Ocp(T=1)

            # Only variables
            ocp = Ocp(T=1)

            x = ocp.variable(grid='bspline',order=order)
            ocp.add_objective(ocp.sum(sumsqr(x-ocp.t),include_last=True))
            ocp.solver('ipopt')

            ocp.method(method)

            # Solve
            sol = ocp.solve()
            [t,xs] = sol.sample(x, grid='control')
            if order==1:
              assert_array_almost_equal(t, xs)
            if order==0:
              assert_array_almost_equal(t[:-2], xs[:-2])
              assert_array_almost_equal(xs[-2:],0.95)


    def test_grid_gist(self):
      for order in range(3):
        ocp = Ocp(T=1)
        x = ocp.control(2,order=order)
        ocp.add_objective(ocp.sum(sumsqr(x-ocp.t),include_last=True))
        ocp.solver('ipopt')
        ocp.method(SplineMethod(N=10))
        sol = ocp.solve()

        [_,xs] = sol.sample(x, grid='gist')
        self.assertEqual(xs.shape[0], 10+order)
        [_,xs2] = sol.sample(7*x+1.7, grid='gist')
        assert_array_almost_equal(7*xs+1.7,xs2)
        [_,xs2] = sol.sample(vertcat(3*x[0]+1.3,x[1]+1.7), grid='gist')
        assert_array_almost_equal(xs[:,0]*3+1.3,xs2[:,0])
        assert_array_almost_equal(xs[:,1]+1.7,xs2[:,1])
        [_,xs2] = sol.sample(vertcat(x[1]+1.3,3*x[0]+1.7), grid='gist')
        assert_array_almost_equal(xs[:,1]+1.3,xs2[:,0])
        assert_array_almost_equal(xs[:,0]*3+1.7,xs2[:,1])
        [_,xs] = ocp.sample(x, grid='gist')
        self.assertEqual(xs.shape[1], 10+order)

        ocp = Ocp(T=1)
        x = ocp.variable(2,grid='bspline',order=order)
        ocp.add_objective(ocp.sum(sumsqr(x-ocp.t),include_last=True))
        ocp.solver('ipopt')
        ocp.method(SplineMethod(N=10))
        sol = ocp.solve()

        [_,xs] = sol.sample(x, grid='gist')
        self.assertEqual(xs.shape[0], 10+order)
        [_,xs2] = sol.sample(7*x+1, grid='gist')
        assert_array_almost_equal(7*xs+1,xs2)
        [_,xs] = ocp.sample(x, grid='gist')
        self.assertEqual(xs.shape[1], 10+order)

        ocp = Ocp(T=1)
        x = ocp.parameter(2,grid='bspline',order=order)
        DM.rng(1)
        ocp.set_value(x, DM.rand(2,10+order))
        ocp.add_objective(ocp.sum(sumsqr(x),include_last=True))
        ocp.solver('ipopt')
        ocp.method(SplineMethod(N=10))
        sol = ocp.solve()

        [_,xs] = sol.sample(x, grid='gist')
        self.assertEqual(xs.shape[0], 10+order)
        [_,xs2] = sol.sample(7*x+1, grid='gist')
        assert_array_almost_equal(7*xs+1,xs2)
        [_,xs] = ocp.sample(x, grid='gist')
        self.assertEqual(xs.shape[1], 10+order)

    def test_der_signals(self):
      for grid in [None,'inf']:
        # Only variables
        ocp = Ocp(T=1.3)

        x = ocp.variable(grid='bspline',order=2)
        ocp.subject_to(ocp.at_t0(x)==0)
        ocp.add_objective(-ocp.at_tf(x))

        ocp.subject_to(ocp.der(x)<=1.1,grid=grid)
        ocp.solver('ipopt')
        ocp.method(SplineMethod(N=10))
        sol = ocp.solve()

        assert_array_almost_equal(sol.sample(ocp.der(x),grid='gist')[1], 1.1)

        self.assertAlmostEqual(sol.value(ocp.at_tf(x)),1.3*1.1)

    def test_p_bspline(self):
      ocp = Ocp(T=1)
      x = ocp.control(order=2)

      p = ocp.parameter(grid='bspline',order=2)
      ocp.add_objective(ocp.sum(sumsqr(x-p),include_last=True))
      DM.rng(1)
      coeff = DM.rand(1,12)
      ocp.set_value(p, coeff)
      ocp.solver('ipopt')
      ocp.method(SplineMethod(N=10))

      # Solve
      sol = ocp.solve()
      [_,xs] = sol.sample(x, grid='gist')
      #print(xs.shape)
      #print(np.array(coeff).squeeze().shape)
      #assert_array_almost_equal(np.array(coeff).squeeze(), xs)

      [_,ds] = sol.sample(x-p, grid='control')
      assert_array_almost_equal(ds, 0)

    def test_basic(self):
        for T in [1, 3.2]:
            for M in [1, 2]:
                for u_max in [1, 2]:
                    for t0 in [0, 1]:
                        for x0 in [0, 1]:
                            for method in [MultipleShooting(N=4,M=M,intg='rk'),
                                           MultipleShooting(N=4,M=M,intg='cvodes'),
                                           MultipleShooting(N=4,M=M,intg='idas'),
                                           DirectCollocation(N=4,M=M),
                                           SingleShooting(N=4,M=M,intg='rk'),
                                           SplineMethod(N=4)]:
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

    def test_parametric_time(self):
        ocp = Ocp()

        T = ocp.parameter()
        ocp.set_T(T)
        ocp.set_value(T, 3)
        x = ocp.state()
        u = ocp.control()

        ocp.set_der(x, u)

        ocp.subject_to(u <= 1)
        ocp.subject_to(-1 <= u)

        ocp.add_objective(ocp.at_tf(x))
        ocp.subject_to(ocp.at_t0(x) == 0)

        ocp.solver('ipopt')

        ocp.method(MultipleShooting())

        sol = ocp.solve()

        ts, xs = sol.sample(x, grid='control')
        self.assertAlmostEqual(xs[0], 0)
        self.assertAlmostEqual(xs[-1], -3,places=6)
        self.assertAlmostEqual(ts[-1], 3,places=6)

        ocp.set_value(T, 1)
        sol = ocp.solve()

        ts, xs = sol.sample(x, grid='control')
        self.assertAlmostEqual(xs[-1], -1,places=6)
        self.assertAlmostEqual(ts[-1], 1,places=6)
  
    def test_basic_time_free(self):
        xf = 2
        for t0 in [0, 1]:
            for x0 in [0, 1]:
                for b in [1.0, 2.0]:
                    for method in [MultipleShooting(N=4, intg='rk'), MultipleShooting(N=4, intg='cvodes'), MultipleShooting(N=4, intg='idas'), DirectCollocation(N=4), SplineMethod(N=4)]:
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
                    for method in [MultipleShooting(N=4, intg='rk'), MultipleShooting(N=4, intg='cvodes'), MultipleShooting(N=4, intg='idas'), DirectCollocation(N=4), SplineMethod(N=4)]:
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
      for stage_method in [SingleShooting(),MultipleShooting(), DirectCollocation(), SplineMethod()]:
        ocp, x, u = integrator_control_problem(x0=None,stage_method=stage_method)

        ocp = Ocp(T=1)

        x = ocp.state()
        u = ocp.control()
        if "SplineMethod" in str(stage_method):
           ocp.set_der(x, u)
        else:
          p = ocp.parameter()
          ocp.set_value(p,1)
          ocp.set_der(x, p*u)

        ocp.subject_to(-1 <= (u <= 1))

        ocp.add_objective(ocp.at_tf(sin(x))+ocp.integral(u**2))

        ocp.solver('ipopt')

        ocp.method(stage_method)

        ocp.set_initial(x, -pi/2+0.1)
        sol = ocp.solve()
        ts, xs = sol.sample(x, grid='control')
        self.assertAlmostEqual(xs[0], -pi/2, places=6)
        ocp.set_initial(x, 2*pi-pi/2+0.1)
        sol = ocp.solve()
        ts, xs = sol.sample(x, grid='control')
        self.assertAlmostEqual(xs[0], 2*pi-pi/2, places=6)

      for stage_method in [SingleShooting(),MultipleShooting(), DirectCollocation(), SplineMethod()]:
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
        for method in [SplineMethod(N=4)]:
            ocp.method(method)
            with self.assertRaisesRegex(Exception,"Time dependent variables not supported in SplineMethod"):
              sol = ocp.solve()


    def test_variables(self):
        N = 10
        ocp = Ocp(t0=2*pi,T=10)
        p = ocp.parameter(grid='control')
        v = ocp.variable(grid='control')
        x = ocp.state()
        ocp.set_der(x, 0)
        ocp.subject_to(ocp.at_t0(x)==0)

        ts = linspace(0, 10, N+1)[:-1]

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

    def test_variables_plus(self):
        N = 10
        ocp = Ocp(t0=2*pi,T=10)
        p = ocp.parameter(grid='control',include_last=True)
        v = ocp.variable(grid='control',include_last=True)
        x = ocp.state()
        ocp.set_der(x, 0)
        ocp.subject_to(ocp.at_t0(x)==0)

        ts = linspace(0, 10, N+1)

        ocp.add_objective(ocp.sum(sin(v-p)**2,include_last=True))
        ocp.method(MultipleShooting(N=N))
        ocp.solver('ipopt')
        ocp.set_value(p, ts)
        ocp.set_initial(v, ts)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')

        assert_array_almost_equal(xs, ts)
        ocp.set_initial(v, 0.1+2*pi+ts)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')
        assert_array_almost_equal(xs, 2*pi+ts)
        ocp.set_initial(v, 0.1+ocp.t)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')
        assert_array_almost_equal(xs, 2*pi+ts)
        ocp.set_initial(v, 0.1+2*pi)
        sol = ocp.solve()
        _, xs = sol.sample(v, grid='control')
        with self.assertRaises(AssertionError):
          assert_array_almost_equal(xs, 2*pi+ts)

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
                DirectCollocation(),
                #SplineMethod()
                ]:
            ocp.method(method)
            sol = ocp.solve()
            ts, xs = sol.sample(f,grid='control')
            I = lambda t: t**3/3-t**2/2*t0
            x_ref = I(t0+T)-I(t0)

            assert_array_almost_equal(xs[-1],x_ref)

    def test_quad(self):
    
    
        for M in [1,2]:
            methods = [SingleShooting(N=3,M=M),
                           MultipleShooting(N=3,M=M),
                           DirectCollocation(N=3,M=M)]
        
            for method in methods:
                ocp = Ocp(T=6)
                
                u = ocp.control()
                q = ocp.state(quad=True)
                q2 = ocp.state(quad=True)
                ocp.set_der(q, u)
                ocp.set_der(q2, u**2)

                ocp.add_objective(ocp.integral(sumsqr(u-sin(ocp.t)))+ocp.at_tf(q2)**2)
                ocp.subject_to(ocp.at_tf(q)==2)
                ocp.solver('ipopt')
                ocp.method(method)
                sol = ocp.solve()
                
                for kwargs in [{"grid":"control"},{"grid":"integrator"},{"grid":"integrator","refine":4}]:
                    refine = kwargs["refine"] if "refine" in kwargs else None
                    
                    if isinstance(method,DirectCollocation):
                        continue
          
                    [_,usol] = sol.sample(u,**kwargs)
                    [_,qsol] = sol.sample(q,**kwargs)
                    
                    q_ref = np.array(vertcat(0,2*np.cumsum(usol[:-1]))).squeeze()
                    if refine is not None:
                        q_ref = q_ref/refine
                    if kwargs["grid"]=="integrator":
                        q_ref = q_ref/M

                    assert_array_almost_equal(qsol,q_ref)

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

        f = ocp.to_function('f',[ocp.value(pp),ocp.sample(p,grid='control')[1], ocp.sample(v,grid='control')[1], ocp.sample(u,grid='control-')[1]],[ocp.sample(p,grid='control')[1]])

        print(f(1,1,2,3))

    def test_to_function_init(self):
        N = 10
        method = DirectCollocation(N=N,M=2,scheme="legendre")
        ocp, x1, x2, u = vdp_dae(method)
        alg = ocp.z


        ocp.solver("ipopt",{"ipopt.tol":1e-12,"dump_in":True})
      
 
        DM.rng(1)
        
        alg0 = np.round(DM.rand(N+1,1),decimals=3)
        x0 = np.round(DM.rand(2,N+1),decimals=3)


        ocp.set_initial(alg, MX(alg0)[ocp.t])
        ocp.set_initial(ocp.x, MX(x0)[:,ca.floor(ocp.t)])

        sol = ocp.solve()
        x0_ref = DM.from_file("solver.000000.in.x0.mtx")

        ocp, x1, x2, u = vdp_dae(method)
        alg = ocp.z


        ocp.solver("ipopt",{"ipopt.tol":1e-12,"dump_in":True})
    
        _,states = ocp.sample(ocp.x,grid='control')

        f = ocp.to_function('F',[states,"z"],[states])
        f(x0,alg0)
      
        x0 = DM.from_file("solver.000000.in.x0.mtx")


        assert_array_almost_equal(x0,x0_ref)




    def test_no_solver(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      p = ocp.parameter()
      ocp.set_der(x, u*p)
      ocp.method(MultipleShooting(N=3))
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

    def test_variable_only(self):
      ocp = Ocp()
      v = ocp.variable()
      ocp.subject_to(v>=5)
      ocp.add_objective(v**2)
      ocp.solver("ipopt")
      sol = ocp.solve()
      self.assertAlmostEqual(sol.value(v),5)

      ocp = Ocp()
      v = ocp.variable()
      ocp.add_objective((v-3)**2)
      ocp.solver("ipopt")
      sol = ocp.solve()
      self.assertAlmostEqual(sol.value(v),3)

      ocp = Ocp()
      v = ocp.variable()
      p = ocp.parameter()
      ocp.add_objective((v-p)**2)
      ocp.solver("ipopt")
      ocp.set_value(p,3)
      sol = ocp.solve()
      ocp.set_value(p,4)
      sol = ocp.solve()
      self.assertAlmostEqual(sol.value(v),4)

      ocp = Ocp()
      v = ocp.variable()
      p = ocp.parameter()
      ocp.subject_to(v>=p)
      ocp.add_objective(v**2)
      ocp.solver("ipopt")
      ocp.set_value(p,5)
      sol = ocp.solve()
      self.assertAlmostEqual(sol.value(v),5)

      ocp.save('test.rockit')

      ocp2 = Ocp.load("test.rockit")
      sol2 = ocp2.solve()
      self.assertAlmostEqual(sol2.value(ocp2.v),5)

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

    def test_set_initial_alg(self):
    
      for method in [SingleShooting(N=10,intg='idas'),MultipleShooting(N=10,intg='idas'),DirectCollocation(N=10)]:
        ocp = Ocp()
        x = ocp.state()
        zx = ocp.algebraic()
        y = ocp.state()
        zy = ocp.algebraic()
        
        ocp.add_alg(sin(zx))
        ocp.add_alg(ca.cos(zy))
        
        u = ocp.control()
        
        ocp.set_der(x, u/zx)
        ocp.set_der(y, u/zy)
        
        ocp.subject_to(ocp.at_t0(x)==0)
        ocp.subject_to(ocp.at_t0(y)==0)
        
        ocp.add_objective(ocp.sum(u**2))
        
        ocp.subject_to(ocp.at_tf(x)==1)
        
        ocp.method(method)
        ocp.solver("ipopt")
        
        with self.assertRaises(Exception):
        	sol = ocp.solve()
        
        ocp.set_initial(zy, pi/2-0.1)
        ocp.set_initial(zx, pi-0.1)
        
        sol = ocp.solve()
        
        np.testing.assert_allclose(sol.sample(zy,grid='control')[1],pi/2)
        np.testing.assert_allclose(sol.sample(zx,grid='control')[1],pi)
      
      for method in [MultipleShooting(N=4,intg='idas'),SingleShooting(N=4,intg='idas'),DirectCollocation(N=4,scheme='legendre')]:
        ocp = Ocp(T=4)
        x = ocp.state()
        zx = ocp.algebraic()
        y = ocp.state()
        zy = ocp.algebraic()
        
        ocp.add_alg(sin(zx))
        ocp.add_alg(ca.cos(zy))
        
        ocp.set_initial(zy, pi/2+pi*ca.floor(ocp.t))
        ocp.set_initial(zx, pi+pi*ca.floor(ocp.t))
        
        u = ocp.control()
        
        ocp.set_der(x, u/zx)
        ocp.set_der(y, u/zy)
        
        ocp.subject_to(ocp.at_t0(x)==0)
        ocp.subject_to(ocp.at_t0(y)==0)
        
        ocp.add_objective(ocp.sum(u**2))
        
        ocp.subject_to(ocp.at_tf(x)==1)
        
        ocp.method(method)
        ocp.solver("ipopt")
        
        sol = ocp.solve()
        
        print(sol.sample(zy,grid='control')[1])
        print(sol.sample(zx,grid='control')[1])
        print(pi/2+pi*np.linspace(0,4,4+1))

        if "MultipleShooting" in str(method):
          # Note the first sample of z is based on discrete_system Zi[0] and Zi is the collection of integrator outputs zf.
          # So for M=1, sampled z is same for k=0 and k=1.
          

          # Furthermore, the first zf here gets initial guess (pi/2), leading to pi/2 solution
          np.testing.assert_allclose(sol.sample(zy,grid='control')[1],[pi/2,pi/2,pi/2+pi,pi/2+2*pi,pi/2+3*pi])
          np.testing.assert_allclose(sol.sample(zx,grid='control')[1],[pi,pi,pi+pi,pi+2*pi,pi+3*pi])
        elif "SingleShooting" in str(method):
          # In single shooting, only the initial guess (pi/2) is used
          np.testing.assert_allclose(sol.sample(zy,grid='control')[1],pi/2)
          np.testing.assert_allclose(sol.sample(zx,grid='control')[1],pi)
        elif "DirectCollocation" in str(method):
          legendre = [pi/2]*4+[pi/2+pi]*4+[pi/2+2*pi]*4+[pi/2+3*pi]*4
          np.testing.assert_allclose(sol.sample(zy,grid='integrator_roots')[1],legendre)
          # Note sure if the answer below is desirable
          np.testing.assert_allclose(sol.sample(zy,grid='control')[1],[pi/2,pi/2+pi,pi/2+2*pi,pi/2+3*pi,pi/2+3*pi])

      
    def test_control_set_der(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      with self.assertRaisesRegex(Exception, "You used set_der on a non"):
        ocp.set_der(u, 2)


    def test_add_objective_signal(self):
      ocp = Ocp()
      x = ocp.state()
      u = ocp.control()
      p = ocp.parameter()
      y = MX.sym('y')
      ocp.set_der(x, u*p)
      ocp.solver('ipopt')
      with self.assertRaisesRegex(Exception, "An objective cannot be a signal"):
        ocp.add_objective(x)

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

    def test_augmented_set_initial(self):
      ocp = Ocp(t0=FreeTime(0),T=FreeTime(2))

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
      ocp.subject_to(ocp.t0 == 0)
      ocp.subject_to(ocp.tf == 2)

      ocp.subject_to(-1 <= (u <= 1))
      ocp.solver('ipopt',{"ipopt.tol":1e-12})

      ocp.method(MultipleShooting(N=4))

      sol = ocp.solve()
      ocp.set_initial(ocp.t0, 2)
      ocp.set_initial(ocp.T, 2)
      ocp.solve()
      
      
    def test_der_t(self):
      ocp = Ocp()

      r = evalf(ocp.der(ocp.t))
      self.assertEqual(r,1)

    def test_subject_to_ocp_t(self):
      for method in [MultipleShooting(N=10),SingleShooting(N=10),DirectCollocation(N=10),SplineMethod(N=10)]:
        ocp = Ocp(T=10)
        p = ocp.state()
        v = ocp.state()
        u = ocp.control()
        ocp.set_der(p, v)
        ocp.set_der(v, u)
        ocp.add_objective(-ocp.at_tf(p))

        ocp.subject_to(v<= 2+sin(ocp.t))

        ocp.subject_to(ocp.at_t0(p)==0)

        ocp.method(method)
        ocp.solver('ipopt')

        sol = ocp.solve()

        [ts,vsol] = sol.sample(v,grid='control')
        np.testing.assert_allclose(vsol, 2+sin(ts), atol=1e-6)

      for method in [MultipleShooting(N=10),SingleShooting(N=10),DirectCollocation(N=10),SplineMethod(N=10)]:
        ocp = Ocp(T=FreeTime(10))
        p = ocp.state()
        v = ocp.state()
        u = ocp.control()
        ocp.set_der(p, v)
        ocp.set_der(v, u)
        ocp.add_objective(ocp.T)

        ocp.subject_to(v<= 2+sin(ocp.t))

        ocp.subject_to(ocp.at_t0(p)==0)
        ocp.subject_to(ocp.at_tf(p)==5)

        ocp.method(method)
        ocp.solver('ipopt')

        sol = ocp.solve()

        [ts,vsol] = sol.sample(v,grid='control')
        np.testing.assert_allclose(vsol, 2+sin(ts), atol=1e-6)

    def test_exploiting_constraints(self):
        # Get as far as possible with a speed limit
        # If we are not careful with enforcing the speed limit, we may end up in cheating solutions that oscillate
        for N in [9,10]:
          for method in [SplineMethod(N=N),MultipleShooting(N=N),SingleShooting(N=N),DirectCollocation(N=N),SplineMethod(N=N)]:
            escape = N % 2 == 1

            for refine_subject_to in [True,False]:
              if refine_subject_to and not isinstance(method,SplineMethod): continue
              ocp = Ocp(T=10)
              p = ocp.control(order=3)
              v = ocp.der(p)
              ocp.add_objective(-ocp.at_tf(p))

              ocp.subject_to(v<=3,grid='inf')

              kwargs = {}
              if refine_subject_to:
                kwargs["refine"] = 2

              ocp.subject_to(v<=2, **kwargs)

              ocp.subject_to(ocp.at_t0(p)==0)

              ocp.method(method)
              ocp.solver('ipopt')

              sol = ocp.solve()

              [ts,vsol] = sol.sample(v,grid='control')
              [tsf,vfsol] = sol.sample(v,grid='control' if isinstance(method, SplineMethod) else 'integrator',refine=10)
              if escape and not refine_subject_to:
                print(method)
                np.testing.assert_allclose(vsol, 2, atol=1e-6)
                print(vfsol)
                # Cheats
                self.assertTrue(vfsol[5]>=2.3)
              else:
                np.testing.assert_allclose(vfsol, 2, atol=1e-6)
          #import pylab as plt
          #plt.plot(ts,vsol,'o')
          #plt.plot(tsf,vfsol,'-')
          #plt.show()
          #np.testing.assert_allclose(vsol, 2+sin(ts), atol=1e-6)

    def test_subject_to_group_refine(self):
        ocp = Ocp(T=10)
        p = ocp.control(order=3)
        v = ocp.der(p)
        ocp.add_objective(-ocp.at_tf(p))

        ocp.subject_to(v<=3,grid='inf')

        ocp.subject_to(v<=2)

        ocp.subject_to(ocp.at_t0(p)==0)

        ocp.method(SplineMethod(N=9))
        ocp.solver('ipopt')

        sol = ocp.solve()
        size_origin = ocp.jacobian().size1()

        ocp = Ocp(T=10)
        p = ocp.control(order=3)
        v = ocp.der(p)
        ocp.add_objective(-ocp.at_tf(p))

        ocp.subject_to(v<=3,grid='inf')

        ocp.subject_to(v<=2, refine=2)

        ocp.subject_to(ocp.at_t0(p)==0)

        ocp.method(SplineMethod(N=9))
        ocp.solver('ipopt')

        sol = ocp.solve()

        [ts,vsol] = sol.sample(v,grid='control')
        [tsf,vfsol] = sol.sample(v,grid='control',refine=10)
        size_normal = ocp.jacobian().size1()
        self.assertTrue(size_normal>size_origin)

        ocp = Ocp(T=10)
        p = ocp.control(order=3)
        v = ocp.der(p)
        ocp.add_objective(-ocp.at_tf(p))

        ocp.subject_to(v<=3,grid='inf')

        ocp.subject_to(v<=2, refine=2, group_refine=LseGroup(margin_abs=0.1))

        ocp.subject_to(ocp.at_t0(p)==0)

        ocp.method(SplineMethod(N=9))
        ocp.solver('ipopt')

        sol = ocp.solve()

        [ts,vsol] = sol.sample(v,grid='control')
        [tsf,vfsol] = sol.sample(v,grid='control',refine=10)

        # Check that grouping satisfies the constraint
        np.testing.assert_array_less(vfsol, 2+1e-8)

        size_lse = ocp.jacobian().size1()
        self.assertEqual(size_lse,size_origin)

        # Lower bound active

        ocp = Ocp(T=10)
        p = ocp.control(order=3)
        v = ocp.der(p)
        ocp.add_objective(-ocp.at_tf(p))

        nv = -v

        ocp.subject_to(v<=3,grid='inf')

        ocp.subject_to(-2<=nv, refine=2, group_refine=LseGroup(margin_abs=0.1))

        ocp.subject_to(ocp.at_t0(p)==0)

        ocp.method(SplineMethod(N=9))
        ocp.solver('ipopt')

        sol = ocp.solve()
        [ts,vsol] = sol.sample(v,grid='control')
        [tsf,vfsol2] = sol.sample(v,grid='control',refine=10)

        size_lse2 = ocp.jacobian().size1()
        np.testing.assert_allclose(vfsol, vfsol2, atol=1e-6)
        self.assertEqual(size_lse,size_lse2)


    def test_initial_spline(self):
        for order in range(1,4):
          for power in range(2): # Does not work for 2; set_initial is correctly setting the control points; but a spline does not pass exactly through control points
            t0 = 1
            T = 5
            ocp = Ocp(t0=1,T=T)
            x = ocp.control(order=order)
            v0 = 3
            x0 = 2
            v = ocp.der(x)
            ocp.subject_to(ocp.at_t0(x)==0)
            ocp.set_initial(x, x0 + v0*ocp.t**power)
            ocp.method(SplineMethod(N=4))
            ocp.solver('ipopt',{'ipopt.max_iter':0})
            print(order,power)
            sol = ocp.solve_limited()

            [ts,xsol] = sol.sample(x,grid='control')
            np.testing.assert_allclose(xsol, x0 + v0*ts**power, atol=1e-6)

            [tsf,xfsol] = sol.sample(x,grid='control',refine=10)
            [_,vfsol] = sol.sample(v,grid='control',refine=10)

            np.testing.assert_allclose(xfsol, x0 + v0*tsf**power, atol=1e-6)
            np.testing.assert_allclose(vfsol, power*v0*tsf**(power-1), atol=1e-6)


    def test_samplingmethod_architecture(self):
        for method in [MultipleShooting(N=4),SingleShooting(N=4),DirectCollocation(N=4),SplineMethod(N=4)]:
            print(method)
            (ocp, p, v, a) = bang_bang_problem(method)
            alpha = ocp.parameter()
            ocp.set_value(alpha,3)
            ocp.add_objective(alpha*ocp.integral(a**2))
            [_,psol] = ocp.sample(p)
            
            syms = symvar(psol)
            self.assertTrue(np.all(["opti" in e.name() for e in syms]))

            [_,psol] = ocp.sample(alpha*p)
            
            syms = symvar(psol)
            self.assertTrue(np.all(["opti" in e.name() for e in syms]))

    def test_pickling(self):
      for method in [MultipleShooting(N=4),SingleShooting(N=4),DirectCollocation(N=4),SplineMethod(N=4)]:
        (ocp, p, v, a) = bang_bang_problem(method)

        sol = ocp.solve()
        [tr,xr] = sol.sample(ocp.x,grid='control')
        ocp.save("foo.rockit")
        ocp2 = Ocp.load("foo.rockit")

        sol2 = ocp2.solve()
        [t2,x2] = sol2.sample(ocp2.x,grid='control')

        np.testing.assert_allclose(tr, t2,atol=1e-16)
        np.testing.assert_allclose(xr, x2,atol=1e-16)

    def test_set_initial_bspline_variable(self):

        ocp = Ocp(T=10)

        def R_compress(R):
            return ca.vec(R)


        def R_decompress(Rc):
            return ca.reshape(Rc,3,3)


        Rc = ocp.variable(9, grid='bspline', order=4)
        R = R_decompress(Rc)


        ocp.subject_to(ocp.at_t0(R)==ca.DM.eye(3))

        #ocp.subject_to(ca.vec(R.T @ R-ca.DM.eye(3))==0)

        ocp.method(SplineMethod(N=5))

        ocp.solver("ipopt",{"ipopt.max_iter":0})

        print(ca.densify(R_compress(ca.DM.eye(3))))
        ocp.set_initial(Rc,ca.densify(R_compress(ca.DM.eye(3))))

        sol = ocp.solve_limited()

        for Rc_sol in ca.vertsplit(sol.sample(Rc,grid='control')[1]):
          assert_array_almost_equal(R_decompress(Rc_sol),DM.eye(3))

    def test_variable_bspline(self):
        
        N = 3
        M = 2
        
        for grid in [UniformGrid(),  GeometricGrid(2)]:
          for degree in [1,2,0]:
            ocp = Ocp(T=10)

            # Define 2 states
            x1 = ocp.state()
            x2 = ocp.state()

            # Define 1 control
            u = ocp.control(order=degree)

            # Specify ODE
            ocp.set_der(x1, (1 - x2**2) * x1 - x2 + u)
            ocp.set_der(x2, x1)

            # Lagrange objective
            ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))

            # Path constraints
            ocp.subject_to(-1 <= (u <= 1))
            ocp.subject_to(x1 >= -0.25, grid='integrator')

            # Initial constraints
            ocp.subject_to(ocp.at_t0(x1) == 0)
            ocp.subject_to(ocp.at_t0(x2) == 1)

            # Pick an NLP solver backend
            ocp.solver('ipopt')

            # Pick a solution method
            ocp.method(DirectCollocation(N=N,M=M,grid=grid))
            
            sol = ocp.solve()

            [_,urefsol_control] = sol.sample(u,grid='control')
            [_,urefsol_integrator] = sol.sample(u,grid='integrator')
            [_,urefsol_integrator_refine] = sol.sample(u,grid='integrator',refine=7)
            [_,urefsol_integrator_roots] = sol.sample(u,grid='integrator_roots')


            ocp = Ocp(T=10)

            # Define 2 states
            x1 = ocp.state()
            x2 = ocp.state()

            # Define 1 control
            u = ocp.variable(grid='bspline',order=degree)

            # Specify ODE
            ocp.set_der(x1, (1 - x2**2) * x1 - x2 + u)
            ocp.set_der(x2, x1)

            # Lagrange objective
            ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))

            # Path constraints
            ocp.subject_to(-1 <= (u <= 1))
            ocp.subject_to(x1 >= -0.25, grid='integrator')

            # Initial constraints
            ocp.subject_to(ocp.at_t0(x1) == 0)
            ocp.subject_to(ocp.at_t0(x2) == 1)

            # Pick an NLP solver backend
            ocp.solver('ipopt')

            # Pick a solution method
            ocp.method(DirectCollocation(N=N,M=M,grid=grid))
            
            sol = ocp.solve()

            [_,usol_control] = sol.sample(u,grid='control')
            #[_,usol_integrator] = sol.sample(u,grid='integrator')
            [_,usol_integrator_refine] = sol.sample(u,grid='integrator',refine=7)
            #[_,usol_integrator_roots] = sol.sample(u,grid='integrator_roots')

            assert_array_almost_equal(urefsol_control,usol_control)
            #assert_array_almost_equal(urefsol_integrator,usol_integrator)
            assert_array_almost_equal(urefsol_integrator_refine,usol_integrator_refine)
            #assert_array_almost_equal(urefsol_integrator_roots,usol_integrator_roots)
            # Set initial



if __name__ == '__main__':
    unittest.main()


