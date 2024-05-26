from pylab import *
import unittest

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, Stage, external_method
import numpy as np
from casadi import kron, DM
import casadi as ca
from rockit.casadi_helpers import AutoBrancher

class AcadosTests(unittest.TestCase):

    def test_time_dependence(self):
      for ab in AutoBrancher():

        t0_free = ab.branch() # False
        T_free = ab.branch() # True

        ocp = Ocp(t0=FreeTime(0) if t0_free else 0.03,T=FreeTime(1.0) if T_free else 1.5)

        if t0_free:
          ocp.subject_to(ocp.t0 == 0.03)

        p = ocp.state()
        v = ocp.state()

        u = ocp.control()

        ocp.set_der(p,v)
        if ab.branch():
          ocp.set_der(v,u+ocp.t)
        else:
          ocp.set_der(v,2*u) # here

        ocp.add_objective(ocp.T)
        #ocp.add_objective(ocp.integral(v**2+u**2))
        if not T_free:
          ocp.add_objective(ocp.integral(u**2))
        ocp.add_objective(ocp.at_tf(v-1)**2)

        case = ab.branch(range(6))
        if case==0:
          ocp.subject_to(-1<= (u <= 1))
        elif case==1:
          ocp.subject_to(-1<= (1.2*u <= 1),include_last=False)
        elif case==2:
          ocp.subject_to(-1.3<= (u-ocp.t <= 1),include_last=False)
        elif case==3:
          ocp.subject_to(-1.3<= (u <= 1+ocp.t),include_last=False)
        elif case==4:
          ocp.subject_to(-1.3-ocp.t <= (u<= 1.1+ocp.t),include_last=False)
        elif case==5:
          ocp.subject_to(-1<= u)
          ocp.subject_to(u <= 1+ocp.t,include_last=False)

        ocp.subject_to(ocp.at_t0(p)==0)
        ocp.subject_to(ocp.at_t0(v)==0)

        ocp.subject_to(ocp.at_tf(p)==1)

        ocp.solver('ipopt',{"ipopt.tol":1e-10})

        ocp.set_initial(u,1)
        ocp.set_initial(p, ocp.t)

        # Obtain reference solution
        N = 5
        ocp.method(MultipleShooting(N=N))
        sol = ocp.solve()

        signals = [("p",p),("v",v),("u",u),("t",ocp.t)]
        values = [("T",ocp.T)]#,("T1",ocp.at_t0(ocp.t)),("T2",ocp.at_tf(ocp.t))]

        ref = {}
        for k, expr in signals:
          ref[k] = sol.sample(expr,grid='control')[1]
        for k, expr in values:
          ref[k] = sol.value(expr)

        [ref_t,_] = sol.sample(signals[0][1],grid='control')

        for method in [external_method('acados', N=N,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=N,tol=1e-8)]:




          ocp.method(method)
          sol = ocp.solve()

          print("analysis here")
          print(ocp.T)
          print(sol.value(ocp.T))

          sold = {}
          for k, expr in signals:
            sold[k] = sol.sample(expr,grid='control')[1]
          for k, expr in values:
            sold[k] = sol.value(expr)

          [sol_t,_] = sol.sample(signals[0][1],grid='control')

          tolerance = 1e-6

          for k, expr in signals:
            print(k+"ref",ref[k])
            print(k+"sol",sold[k])
          for k, expr in values:
            print(k+"ref",ref[k])
            print(k+"sol",sold[k])
          print("ref_t",ref_t)
          print("sol_t",sol_t)

          for k, expr in signals:
            np.testing.assert_allclose(sold[k], ref[k], atol=tolerance)
          for k, expr in values:
            np.testing.assert_allclose(sold[k], ref[k], atol=tolerance)
          np.testing.assert_allclose(sol_t, ref_t, atol=tolerance)


    def test_modes(self):
      for ab in AutoBrancher():

        ocp = Ocp(T=1.0)

        x      = ocp.state()
        v      = ocp.state()

        F       = ocp.control()

        ocp.set_der(x, v)
        ocp.set_der(v, F)

        if ab.branch():
          ocp.add_objective(ocp.sum(v**2))
        else:
          ocp.add_objective(ocp.integral(v**2))
        ocp.add_objective(5*ocp.at_tf(v)**2)

        ocp.subject_to(ocp.at_t0(x)==0)
        if ab.branch():
          ocp.subject_to(ocp.at_t0(v)==0.0)
        else:
          pass

        ocp.subject_to(ocp.at_tf(x)==5.0)
        ocp.subject_to(F<=15.0)
        ocp.subject_to(v<=7.0,include_first=ab.branch(),include_last=ab.branch())
      
        ocp.method(MultipleShooting(N=4,intg='rk'))

        ocp.solver('ipopt',{"ipopt.tol":1e-10})

        sol = ocp.solve()

        signals = [("x",x),("v",v),("F",F)]
        values = []#("mayer",mayer)]#[("slack",slack)]

        ref = {}
        for k, expr in signals:
          ref[k] = sol.sample(expr,grid='control')[1]
        for k, expr in values:
          ref[k] = sol.value(expr)

        [ref_t,_] = sol.sample(signals[0][1],grid='control')

        print(ref)


        for method in [external_method('acados',N=4,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=2000,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=4,tol=1e-8)]:

          ocp.method(method)
          sol = ocp.solve()

          sold = {}
          for k, expr in signals:
            sold[k] = sol.sample(expr,grid='control')[1]
          for k, expr in values:
            sold[k] = sol.value(expr)

          [sol_t,_] = sol.sample(signals[0][1],grid='control')

          tolerance = 1e-5

          for k, expr in signals:
            print(k+"ref",ref[k])
            print(k+"sol",sold[k])
          for k, expr in values:
            print(k+"ref",ref[k])
            print(k+"sol",sold[k])
          print("ref_t",ref_t)
          print("sol_t",sol_t)

          for k, expr in signals:
            np.testing.assert_allclose(sold[k], ref[k], atol=tolerance)
          for k, expr in values:
            np.testing.assert_allclose(sold[k], ref[k], atol=tolerance)
          np.testing.assert_allclose(sol_t, ref_t, atol=tolerance)

    def test_slacks(self):
      
      obj_store = None

      for ab in AutoBrancher():
        mode = ab.branch(["nominal","perturbed","soft"])

        ocp = Ocp(T=1.0)

        x      = ocp.state()
        v      = ocp.state()

        F       = ocp.control()

        ocp.set_der(x, v)
        ocp.set_der(v, F)

        ocp.add_objective(ocp.sum(v**2))
        ocp.add_objective(5*ocp.at_tf(v)**2)

        ocp.subject_to(ocp.at_t0(x)==0.0)
        ocp.subject_to(ocp.at_t0(v)==0.0)

        constraints = []

        
        constr_type = ab.branch(["simple","linear","nonlinear"])
        if constr_type=="simple":
          constraints.append((lambda e: F <= 15+e,True,{})) # simple
          constraints.append((lambda e: v <= 7+e,True,dict(include_first=False,include_last=False)))  # include_first=True induces failure
          constraints.append((lambda e: ocp.at_tf(x) + e>=5.0,False,{})) # boundary constraint on state
        elif constr_type=="linear":
          constraints.append((lambda e: F <= 15+e,True,{})) # simple
          constraints.append((lambda e: v+0.01*x <= 7+e,True,dict(include_first=True,include_last=False)))  # include_first=True induces failure
          constraints.append((lambda e: ocp.at_tf(x+0.01*v) + e>=5.0,False,{})) # boundary constraint on state
        elif constr_type=="nonlinear":
          constraints.append((lambda e: F <= 15+e,True,{})) # simple
          constraints.append((lambda e: v+0.01*x**2 <= 7+e,True,dict(include_first=False,include_last=False)))  # include_first=True induces failure
          constraints.append((lambda e: ocp.at_tf(x+0.01*v**2) + e>=5.0,False,{}))



        if mode=="nominal":
          for c,_,kwargs in constraints:
            ocp.subject_to(c(0),**kwargs)
        elif mode=="perturbed":
          k = ab.branch(range(len(constraints)))
          for i,(c,_,kwargs) in enumerate(constraints):
            ocp.subject_to(c(0.1 if i==k else 0),**kwargs)
        elif mode=="soft":
          k = ab.branch(range(len(constraints)))
          power = ab.branch([1,2]) # 1-norm or 2-norm?
          for i,(c,signal,kwargs) in enumerate(constraints):
            if i!=k:
              ocp.subject_to(c(0),**kwargs)
            else:
              slack = ocp.variable(grid='control' if signal else '')
              ocp.subject_to(slack>=0)
              ocp.subject_to(c(slack),**kwargs)
              penalty = 0.3*slack**power
              print("signal",signal, penalty)
              ocp.add_objective(ocp.sum(penalty) if signal else penalty)

        ocp.method(MultipleShooting(N=4,intg='rk'))

        ocp.solver('ipopt',{"ipopt.tol":1e-10})

        sol = ocp.solve()

        # Are the constraints we added really active?
        if mode=="nominal":
          obj_store = sol.value(ocp.objective)
        else:
          print("constraint",k)
          with self.assertRaises(Exception):
            np.testing.assert_allclose(sol.value(ocp.objective), obj_store, atol=0.01)

        signals = [("x",x),("v",v),("F",F)]
        values = []#("mayer",mayer)]#[("slack",slack)]

        ref = {}
        for k, expr in signals:
          ref[k] = sol.sample(expr,grid='control')[1]
        for k, expr in values:
          ref[k] = sol.value(expr)

        [ref_t,_] = sol.sample(signals[0][1],grid='control')

        print(ref)

        #raise Exception("")

        for method in [external_method('acados',N=4,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=2000,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=4)]:

          ocp.method(method)
          sol = ocp.solve()

          sold = {}
          for k, expr in signals:
            sold[k] = sol.sample(expr,grid='control')[1]
          for k, expr in values:
            sold[k] = sol.value(expr)

          [sol_t,_] = sol.sample(signals[0][1],grid='control')

          tolerance = 1e-5

          for k, expr in signals:
            print(k+"ref",ref[k])
            print(k+"sol",sold[k])
          for k, expr in values:
            print(k+"ref",ref[k])
            print(k+"sol",sold[k])
          print("ref_t",ref_t)
          print("sol_t",sol_t)

          for k, expr in signals:
            np.testing.assert_allclose(sold[k], ref[k], atol=tolerance)
          for k, expr in values:
            np.testing.assert_allclose(sold[k], ref[k], atol=tolerance)
          np.testing.assert_allclose(sol_t, ref_t, atol=tolerance)

    def test_parameters(self):
        
      def reference(p_global_value,p_local_value):
        ocp = Ocp(t0=0, T=10)

        # Define two scalar states (vectors and matrices also supported)
        x1 = ocp.state()
        x2 = ocp.state()

        # Define one piecewise constant control input
        #  (use `order=1` for piecewise linear)
        u = ocp.control()

        # Compose time-dependent expressions a.k.a. signals
        #  (explicit time-dependence is supported with `ocp.t`)
        e = 1 - x2**2

        # Specify differential equations for states
        #  (DAEs also supported with `ocp.algebraic` and `add_alg`)
        ocp.set_der(x1, e * x1 - x2 + p_global_value*u)
        ocp.set_der(x2, x1)

        # Lagrange objective term: signals in an integrand
        ocp.add_objective(ocp.sum(x1**2 + (x2-p_local_value(ocp.t))**2 + u**2))
        # Mayer objective term: signals evaluated at t_f = t0_+T
        ocp.add_objective(ocp.at_tf(x1**2))

        # Path constraints
        #  (must be valid on the whole time domain running from `t0` to `tf`,
        #   grid options available such as `grid='inf'`)
        ocp.subject_to(x1 >= -0.25)
        ocp.subject_to(-1 <= (u <= 1 ))

        # Boundary constraints
        ocp.subject_to(ocp.at_t0(x1) == 0)
        ocp.subject_to(ocp.at_t0(x2) == 1)

        # Pick an NLP solver backend
        #  (CasADi `nlpsol` plugin):
        ocp.solver('ipopt')

        # Pick a solution method
        method = external_method('acados', N=10,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=10)
        ocp.method(method)

        # Set initial guesses for states, controls and variables.
        #  Default: zero
        ocp.set_initial(x2, 0)                 # Constant
        ocp.set_initial(x1, ocp.t/10)          # Function of time
        ocp.set_initial(u, linspace(0, 0.1, 10)) # Matrix

        # Solve
        sol = ocp.solve()

        return sol.sample(ocp.x,grid='control')[1]
      
      ref = reference(1,lambda t : 1)
      ref_var_global = reference(1.1,lambda t : 1)
      ref_var_local = reference(1,lambda t : 1.1)
      ref_var_local2 = reference(1,lambda t : 1+0.01*t)
      ref_var = reference(1.1,lambda t : 1+0.01*t)

      ocp = Ocp(t0=0, T=10)

      # Define two scalar states (vectors and matrices also supported)
      x1 = ocp.state()
      x2 = ocp.state()

      # Define one piecewise constant control input
      #  (use `order=1` for piecewise linear)
      u = ocp.control()

      p_local = ocp.parameter(grid='control')
      ocp.set_value(p_local,[1+0*i*0.01 for i in range(11)])

      p_global = ocp.parameter()
      ocp.set_value(p_global,1)

      # Compose time-dependent expressions a.k.a. signals
      #  (explicit time-dependence is supported with `ocp.t`)
      e = 1 - x2**2

      # Specify differential equations for states
      #  (DAEs also supported with `ocp.algebraic` and `add_alg`)
      ocp.set_der(x1, e * x1 - x2 + p_global*u)
      ocp.set_der(x2, x1)

      # Lagrange objective term: signals in an integrand
      ocp.add_objective(ocp.sum(x1**2 + (x2-p_local)**2 + u**2))
      # Mayer objective term: signals evaluated at t_f = t0_+T
      ocp.add_objective(ocp.at_tf(x1**2))

      # Path constraints
      #  (must be valid on the whole time domain running from `t0` to `tf`,
      #   grid options available such as `grid='inf'`)
      ocp.subject_to(x1 >= -0.25)
      ocp.subject_to(-1 <= (u <= 1 ))

      # Boundary constraints
      ocp.subject_to(ocp.at_t0(x1) == 0)
      ocp.subject_to(ocp.at_t0(x2) == 1)

      # Pick an NLP solver backend
      #  (CasADi `nlpsol` plugin):
      ocp.solver('ipopt')

      # Pick a solution method
      method = external_method('acados', N=10,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=10)
      ocp.method(method)

      # Set initial guesses for states, controls and variables.
      #  Default: zero
      ocp.set_initial(x2, 0)                 # Constant
      ocp.set_initial(x1, ocp.t/10)          # Function of time
      ocp.set_initial(u, linspace(0, 0.1, 10)) # Matrix
    
      # Solve
      sol = ocp.solve()

      res = sol.sample(ocp.x,grid='control')[1]
      print(res)

      f = ocp.to_function('f',[ocp.sample(ocp.x,grid='control')[1], ocp.sample(u,grid='control-')[1] ],[ocp.sample(ocp.x,grid='control')[1]])

      np.testing.assert_allclose(res, np.array(f(0,0)).T, atol=1e-5)

      np.testing.assert_allclose(ref, np.array(f(0,0)).T, atol=1e-5)

      f = ocp.to_function('f',[p_global, ocp.sample(p_local,grid='control')[1], ocp.sample(ocp.x,grid='control')[1], ocp.sample(u,grid='control-')[1] ],[ocp.sample(ocp.x,grid='control')[1]])

      np.testing.assert_allclose(ref, np.array(f(1,1,0,0)).T, atol=1e-5)
      np.testing.assert_allclose(ref_var_global, np.array(f(1.1,1,0,0)).T, atol=1e-5)
      np.testing.assert_allclose(ref_var_local, np.array(f(1,1.1,0,0)).T, atol=1e-5)
      np.testing.assert_allclose(ref_var_local2, np.array(f(1,ca.hcat([1+i*0.01 for i in range(11)]),0,0)).T, atol=1e-5)

      np.testing.assert_allclose(ref_var, np.array(f(1.1,ca.hcat([1+i*0.01 for i in range(11)]),0,0)).T, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
