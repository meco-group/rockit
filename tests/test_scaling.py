import unittest

# Make available sin, cos, etc
from numpy import *
# Import the project
from rockit import *
import casadi as cs
import glob
import os

intg_options = {"rootfinder": "nlpsol", "rootfinder_options": {"nlpsol":"ipopt","nlpsol_options":{"ipopt.print_level":0,"print_time":False,"ipopt.tol":1e-10}}}

class ScalingTests(unittest.TestCase):


    def test_vdp(self):

        for Method in [lambda **kwargs: SingleShooting(intg='collocation',intg_options=intg_options,**kwargs), lambda **kwargs: MultipleShooting(intg='collocation',intg_options=intg_options,**kwargs), lambda **kwargs:  DirectCollocation(**kwargs)]:

            data_ref = None
            for run_type in ["ref","scaled"]:
                #%%
                # Problem specification
                # ---------------------

                # Start an optimal control environment with a time horizon of 10 seconds
                # starting from t0=0s.
                #  (free-time problems can be configured with `FreeTime(initial_guess)`)
                ocp = Ocp(t0=0, T=10)

                s1 = 1
                s2 = 1
                su = 1
                s = 1
                si = 1
                sz = 1
                
                if run_type=="scaled":
                    s1 = 2
                    s2 = 3
                    su = 5
                    s = 7
                    si = 11
                    sz = 13

                # Define two scalar states (vectors and matrices also supported)
                x1 = ocp.state(scale=1/s1) # [m/s]
                x2 = ocp.state(scale=1/s2) # [um/s]

                # Define one piecewise constant control input
                #  (use `order=1` for piecewise linear)
                u = ocp.control(scale=1/su)

                z = ocp.algebraic(scale=1/sz)
                
                # Compose time-dependent expressions a.k.a. signals
                #  (explicit time-dependence is supported with `ocp.t`)
                ocp.add_alg(sz*z-(1 - 1e-2*(s2*x2)**2),scale=sz)

                # Specify differential equations for states
                #  (DAEs also supported with `ocp.algebraic` and `add_alg`)
                ocp.set_der(x1, (sz*z * (s1*x1) - (s2*x2) + su*u)/s1, scale=1/s1)
                ocp.set_der(x2, s1*x1/s2, scale=1/s2)

                # Lagrange objective term: signals in an integrand
                ocp.add_objective(ocp.integral(si*((s1*x1)**2 + (s2*x2)**2 + (su*u)**2))/si)
                # Mayer objective term: signals evaluated at t_f = t0_+T
                ocp.add_objective(ocp.at_tf((s1*x1)**2))

                # Path constraints
                #  (must be valid on the whole time domain running from `t0` to `tf`,
                #   grid options available such as `grid='inf'`)


                ocp.subject_to(s*s1*x1 >= -0.25*s, scale=s)
                ocp.subject_to(-1*s <= (su*u*s <= 1*s ), scale=s)

                # Boundary constraints
                ocp.subject_to(ocp.at_t0(s1*x1*s) == 0*s, scale=s)
                ocp.subject_to(ocp.at_t0(s2*x2*s) == 1*s, scale=s)

                #%%
                # Solving the problem
                # -------------------

                # Pick an NLP solver backend
                #  (CasADi `nlpsol` plugin):
                ocp.solver('sqpmethod',{"qpsol_options":{"print_problem":True,"dump_in":True,"dump":True},"convexify_strategy":"eigen-reflect"})


                # Pick a solution method
                #  e.g. SingleShooting, MultipleShooting, DirectCollocation
                #
                #  N -- number of control intervals
                #  M -- number of integration steps per control interval
                #  grid -- could specify e.g. UniformGrid() or GeometricGrid(4)
                method = Method(N=4)
                ocp.method(method)

                # Set initial guesses for states, controls and variables.
                #  Default: zero
                ocp.set_initial(u, linspace(0, 1, 4)/su) # Matrix

                # Solve
                sol = ocp.solve()

                f = cs.Function.load("qpsol.casadi")
                data=f.convert_in(f.generate_in("qpsol.000000.in.txt"))
                if run_type=="ref":
                    data_ref = data
                else:
                    for k in data.keys():

                        n = cs.norm_inf(data[k]-data_ref[k])
                        print(k, n)
                        assert n<=1e-7 # stems from integrator accuracy

        for e in glob.glob("qpsol.*"):
            os.remove(e)

if __name__ == '__main__':
    unittest.main()
