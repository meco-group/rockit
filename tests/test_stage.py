import unittest

from rockit import Ocp, DirectMethod, MultipleShooting, SingleShooting, FreeTime, Stage, GeometricGrid
import numpy as np
from casadi import kron, DM, vertcat

from problems import bang_bang_problem
from rockit.spline_method import SplineMethod

class StageTests(unittest.TestCase):

    def test_stage_next(self):
      for method in [MultipleShooting(N=10),SplineMethod(N=10)]:
        (ocp, p, v, u) = bang_bang_problem(method)

        ocp.subject_to(-0.3 <= (ocp.next(u)-u <=0.3)  )
        #ocp.subject_to(-0.3 <= ((ocp.next(u)-u)/(ocp.next(ocp.t) - ocp.t)<=0.3) )

        sol = ocp.solve()

        usol = sol.sample(u,grid='control')[1]

        self.assertAlmostEqual(np.linalg.norm(np.diff(usol,axis=0),np.inf), 0.3, places=5)

    def test_inf_der(self):
      (ocp, p, v, u) = bang_bang_problem(MultipleShooting(N=10))

      ocp.subject_to(-0.3 <= (ocp.inf_der(p) <= 0.3),grid='inf' )

      sol = ocp.solve()

      vsol = sol.sample(v,grid='control')[1]
      self.assertAlmostEqual(np.linalg.norm(vsol,np.inf),0.3, places=5)

    def test_objective_next(self):
      ref = None
      for method in [MultipleShooting(N=10),SplineMethod(N=10)]:
        (ocp, p, v, u) = bang_bang_problem(method)

        #p_ref = ocp.parameter(2, grid='control')
        #ocp.set_value(p_ref, 1)
        
        x = vertcat(p,v)
        dx = ocp.next(x)-x
        ocp.add_objective(ocp.sum(dx.T @ dx))

        sol = ocp.solve()
        if ref is None:
            ref = sol.value(ocp.T)
        else:
            np.testing.assert_allclose(ref,sol.value(ocp.T))


    def test_set_next(self):
      ocp = Ocp(T=10)

      x1 = ocp.state()
      x2 = ocp.state()

      u = ocp.control()

      ocp.set_next(x1, (1 - x2**2) * x1 - x2 + u)
      ocp.set_next(x2, x1)

      ocp.add_objective(ocp.sum(x1**2 + x2**2 + u**2))

      ocp.subject_to(-1 <= (u <= 1))
      ocp.subject_to(x1 >= -0.25)

      ocp.subject_to(ocp.at_t0(x1) == 0)
      ocp.subject_to(ocp.at_t0(x2) == 1)

      ocp.solver('ipopt')

      N = 10
      for M in [1,2]:
        ocp.method(MultipleShooting(N=N,M=M))
        
        sol = ocp.solve()

        x1sol = sol.sample(x1, grid='integrator')[1]
        x2sol = sol.sample(x2, grid='integrator')[1]
        usol = kron(DM.ones(1,M),sol.sample(u, grid='integrator')[1])
        for k in range(N*M):
          self.assertAlmostEqual(x2sol[k+1], x1sol[k])
          self.assertAlmostEqual(x1sol[k+1], (1 - x2sol[k]**2) * x1sol[k] - x2sol[k] + usol[k])

      
    def test_DT(self):

      for system in ['set_next','set_der']:
          ocp = Ocp(T=10)

          x = ocp.state()

          u = ocp.control()
          
          getattr(ocp,system)(x, u)

          ocp.add_objective(ocp.sum(u**2))

          ocp.subject_to(-1 <= (u <= 1))

          ocp.subject_to(ocp.at_t0(x) == 0)
          ocp.subject_to(ocp.at_tf(x) == 1)
          
          for M in [1,2,3]:

              ocp.solver('ipopt')
              ocp.method(MultipleShooting(N=6,M=M,grid=GeometricGrid(2)))
              sol = ocp.solve()

              [ts,DT_controls] = sol.sample(ocp.DT_control,grid='control')
              np.testing.assert_allclose(np.diff(ts),DT_controls[:-1])
              self.assertEqual(DT_controls[-1],DT_controls[-2])
              
              [_,DT_controls] = sol.sample(ocp.DT_control,grid='integrator')
              np.testing.assert_allclose(np.array(kron(np.diff(ts),[1]*M)).squeeze(),DT_controls[:-1])

              [ts,DTs] = sol.sample(ocp.DT,grid='integrator')
              np.testing.assert_allclose(np.diff(ts),DTs[:-1])
              self.assertEqual(DTs[-1],DTs[-2])

              [ts,DTs_control] = sol.sample(ocp.DT,grid='control')
              np.testing.assert_allclose(DTs[::M],DTs_control)

    def test_DT_system(self):
    
      for dt in ["DT_control","DT",None]:

          for system in ['set_next','set_der']:
              ocp = Ocp(T=10)

              x = ocp.state()

              u = ocp.control()
              
              getattr(ocp,system)(x, (1 if dt is None else getattr(ocp,dt))*u)

              ocp.subject_to(-1 <= (u <= 1))

              ocp.subject_to(ocp.at_t0(x) == 0)
              ocp.add_objective(-ocp.at_tf(x))
              
              for N in [5,10]:
                  for M in [1,2]:

                      ocp.solver('ipopt',{"ipopt.print_level":0,"print_time":False})
                      ocp.method(MultipleShooting(N=N,M=M))
                      if system=="set_der" and dt is not None:
                          with self.assertRaisesRegex(Exception, dt):
                            sol = ocp.solve()
                      else:
                        sol = ocp.solve()
                        xf = sol.value(ocp.at_tf(x))
                        if system=='set_next':
                            if dt == "DT_control":
                                DT_control = 10/N
                                self.assertAlmostEqual(xf, DT_control)
                            elif dt == "DT":
                                DT = 10/N/M
                                self.assertAlmostEqual(xf, DT)
                            else:
                                self.assertAlmostEqual(xf, 1)

    def test_initial(self):
        for t0_stage, t0_sol_stage in [(None, 0), (-1, -1), (FreeTime(-1), -1)]:
            for T_stage, T_sol_stage in [(None, 2), (2, 2), (FreeTime(1), 2)]:
                kwargs = {}
                if t0_stage is not None:
                    kwargs["t0"] = t0_stage
                if T_stage is not None:
                    kwargs["T"] = T_stage
                stage = Stage(**kwargs)

                p = stage.state()
                v = stage.state()
                u = stage.control()

                stage.set_der(p, v)
                stage.set_der(v, u)

                stage.subject_to(u <= 1)
                stage.subject_to(-1 <= u)

                #stage.set_initial(p, stage.t)

                stage.add_objective(stage.tf)
                stage.method(MultipleShooting(N=2))

                for t0, t0_sol in ([] if t0_stage is None else [(None, t0_sol_stage)]) + [(-1, -1), (FreeTime(-1), -1)]:
                    for T, T_sol in ([] if T_stage is None else [(None, T_sol_stage)]) + [(2, 2), (FreeTime(1), 2)]:
                        ocp = Ocp()

                        kwargs = {}
                        if t0 is not None:
                            kwargs["t0"] = t0
                        if T is not None:
                            kwargs["T"] = T
                        mystage = ocp.stage(stage, **kwargs)

                        if isinstance(t0,FreeTime) or (t0 is None and isinstance(t0_stage,FreeTime)):
                          print("stage", mystage.t0)
                          ocp.subject_to(mystage.t0 >= t0_sol)

                        ocp.solver('ipopt',{"ipopt.max_iter": 0})
                        mystage.set_initial(p, mystage.t)
                        sol = ocp.solve_limited()


                        t0_num = sol(mystage).value(mystage.t0)
                        T_num = sol(mystage).value(mystage.T)

                        tolerance = 1e-6

                        ps_ref = np.linspace(t0_num, t0_num+T_num, 2+1)
                        ts, ps = sol(mystage).sample(p, grid='control')
                        np.testing.assert_allclose(ps, ps_ref, atol=tolerance)

    def test_stage_cloning_t0_T(self):
        for t0_stage, t0_sol_stage in [(None, 0), (-1, -1), (FreeTime(-1), -1)]:
            for T_stage, T_sol_stage in [(None, 2), (2, 2), (FreeTime(1), 2)]:
                kwargs = {}
                if t0_stage is not None:
                    kwargs["t0"] = t0_stage
                if T_stage is not None:
                    kwargs["T"] = T_stage
                stage = Stage(**kwargs)

                p = stage.state()
                v = stage.state()
                u = stage.control()

                stage.set_der(p, v)
                stage.set_der(v, u)

                stage.subject_to(u <= 1)
                stage.subject_to(-1 <= u)

                stage.add_objective(stage.tf)
                stage.subject_to(stage.at_t0(p) == 0)
                stage.subject_to(stage.at_t0(v) == 0)
                stage.subject_to(stage.at_tf(p) == 1)
                stage.subject_to(stage.at_tf(v) == 0)
                stage.method(MultipleShooting(N=2))

                for t0, t0_sol in ([] if t0_stage is None else [(None, t0_sol_stage)]) + [(-1, -1), (FreeTime(-1), -1)]:
                    for T, T_sol in ([] if T_stage is None else [(None, T_sol_stage)]) + [(2, 2), (FreeTime(1), 2)]:
                        ocp = Ocp()

                        kwargs = {}
                        if t0 is not None:
                            kwargs["t0"] = t0
                        if T is not None:
                            kwargs["T"] = T
                        mystage = ocp.stage(stage, **kwargs)

                        if isinstance(t0,FreeTime) or (t0 is None and isinstance(t0_stage,FreeTime)):
                          print("stage", mystage.t0)
                          ocp.subject_to(mystage.t0 >= t0_sol)

                        ocp.solver('ipopt')

                        sol = ocp.solve()

                        tolerance = 1e-6

                        ts, ps = sol(mystage).sample(p, grid='integrator', refine=10)

                        ps_ref = np.hstack(((0.5*np.linspace(0,1, 10+1)**2)[:-1],np.linspace(0.5,1.5,10+1)-0.5*np.linspace(0,1, 10+1)**2)) 
                        np.testing.assert_allclose(ps, ps_ref, atol=tolerance)

                        ts_ref = t0_sol + np.linspace(0, 2, 10*2+1)

                        ts, vs = sol(mystage).sample(v, grid='integrator', refine=10)
                        np.testing.assert_allclose(ts, ts_ref, atol=tolerance)

                        vs_ref = np.hstack((np.linspace(0,1, 10+1)[:-1],np.linspace(1,0, 10+1))) 
                        np.testing.assert_allclose(vs, vs_ref, atol=tolerance)


                        u_ref = np.array([1.0]*10+[-1.0]*11)
                        ts, us = sol(mystage).sample(u, grid='integrator', refine=10)
                        np.testing.assert_allclose(us, u_ref, atol=tolerance)

if __name__ == '__main__':
    unittest.main()
