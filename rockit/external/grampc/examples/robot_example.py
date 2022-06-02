"""
A rockit implemantation of the robot problem
"""
import rockit as roc
import numpy as np
import casadi as cs

# Start an optimal control environment with a time horizon of 10 seconds
# starting from t0=0s.
#  (free-time problems can be configured with `FreeTime(initial_guess)`)
ocp = roc.Ocp(t0=0, T=1)

# Define scalar states (vectors and matrices also supported)
x1 = ocp.state()
x2 = ocp.state()
x3 = ocp.state()
x4 = ocp.state()
x5 = ocp.state()
x6 = ocp.state()

# Define piecewise constant control inputs
u1 = ocp.control()
u2 = ocp.control()
u3 = ocp.control()
u4 = ocp.control()
u5 = ocp.control()
u6 = ocp.control()

a1 = ocp.parameter()
a2 = ocp.parameter()
a3 = ocp.parameter()
a4 = ocp.parameter()
a5 = ocp.parameter()
a6 = ocp.parameter()

u = cs.vertcat(u1, u2,u3,u4, u5, u6)

# Define initial and terminal state
x0 = cs.vertcat(cs.pi/2, -cs.pi/2, 0, -cs.pi/2, cs.pi/2, 0)
xf = cs.vertcat(-cs.pi/2, cs.pi/2, 0, cs.pi/2, -cs.pi/2, 0)

# Specify differential equations for states
#  (DAEs also supported with `ocp.algebraic` and `add_alg`)
ocp.set_der(x1, u1)
ocp.set_der(x2, u2)
ocp.set_der(x3, u3)
ocp.set_der(x4, u4)
ocp.set_der(x5, u5)
ocp.set_der(x6, u6)

# Lagrange objective term: signals in an integrand
ocp.add_objective(ocp.integral(1 * u.T @ u))
ocp.add_objective(ocp.at_tf(1))

# Define forward kinematics
offset_left = cs.vertcat(0, 0, 0)
offset_right = cs.vertcat(1, 0, cs.pi)

p_L = cs.vertcat(a1*cs.cos(x1+ offset_left[2]) + a2*cs.cos(x1+x2+ offset_left[2]) + a3*cs.cos(x1 + x2 + x3+ offset_left[2]) + offset_left[0],
                 a1*cs.sin(x1+ offset_left[2]) + a2*cs.sin(x1+x2+ offset_left[2]) + a3*cs.sin(x1 + x2 + x3+ offset_left[2]) + offset_left[1],
                 x1 + x2 + x3 + offset_left[2])


p_R = cs.vertcat(offset_right[0] + a4*cs.cos(x4 + offset_right[2]) + a5*cs.cos(x4+x5+offset_right[2]) + a6*cs.cos(x4 + x5 + x6+offset_right[2]),
                 offset_right[1] + a4*cs.sin(x4+offset_right[2]) + a5*cs.sin(x4+x5+offset_right[2]) + a6*cs.sin(x4 + x5 + x6 +offset_right[2]),
                 x4 + x5 + x6 + offset_right[2])

# Define expression for closed kinematic chain
g = p_L - p_R + cs.vertcat(0, 0, cs.pi)



# Path constraints on closed kinematic chain
ocp.subject_to(g == 0)#, include_first=False, include_last=False)

# Box constraints on controls
u_max = cs.vertcat(1, 1, 1, 1, 1, 1)
ocp.subject_to(-u_max <= (u <= u_max ))

# Boundary constraints
ocp.subject_to(ocp.at_t0(x1) == cs.pi/2)
ocp.subject_to(ocp.at_t0(x2) == -cs.pi/2)
ocp.subject_to(ocp.at_t0(x3) == 0)
ocp.subject_to(ocp.at_t0(x4) == -cs.pi/2)
ocp.subject_to(ocp.at_t0(x5) == cs.pi/2)
ocp.subject_to(ocp.at_t0(x6) == 0)

ocp.subject_to(ocp.at_tf(x1) == -cs.pi/2)
ocp.subject_to(ocp.at_tf(x2) == cs.pi/2)
ocp.subject_to(ocp.at_tf(x3) == 0)
ocp.subject_to(ocp.at_tf(x4) == cs.pi/2)
ocp.subject_to(ocp.at_tf(x5) == -cs.pi/2)
ocp.subject_to(ocp.at_tf(x6) == 0)

#%%
# Solving the problem
# -------------------

grampc_options = {}
grampc_options["MaxGradIter"] = 1000
grampc_options["MaxMultIter"] = 1
grampc_options["ShiftControl"] = "off"
grampc_options["LineSearchMax"] =  2.0
grampc_options["LineSearchExpAutoFallback"] = "off"
grampc_options["AugLagUpdateGradientRelTol"] = 1.0
grampc_options["ConstraintsAbsTol"] = 1e-4
grampc_options["PenaltyMax"] = 1e4
grampc_options["PenaltyMin"] = 50.0
grampc_options["PenaltyIncreaseFactor"] = 1.1
grampc_options["PenaltyDecreaseFactor"] = 1.0
grampc_options["ConvergenceCheck"] = "on"
grampc_options["ConvergenceGradientRelTol"] = 1e-6
method = roc.external_method('grampc',N=10,M=1,grampc_options=grampc_options)


# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt', {"error_on_fail":False, 'ipopt':{"max_iter": 1000, 'hessian_approximation':'exact', 'limited_memory_max_history' : 5, 'print_level':5}})


# Pick a solution method
N=40
#method = roc.MultipleShooting(N=N, intg='rk')
method = roc.external_method('grampc')
ocp.method(method)

# Set initial guesses for states, controls and variables.
#  Default: zero
ocp.set_value(a1, 0.5)
ocp.set_value(a2, 0.3)
ocp.set_value(a3, 0.2)
ocp.set_value(a4, 0.5)
ocp.set_value(a5, 0.3)
ocp.set_value(a6, 0.2)

dist = np.linspace(0,1, N+1)
x1_0 = cs.pi/2
x2_0 = -cs.pi/2
x3_0 = 0
x4_0 = -cs.pi/2
x5_0 = cs.pi/2
x6_0 = 0

x1_f = -cs.pi/2
x2_f = cs.pi/2
x3_f = 0
x4_f = cs.pi/2
x5_f = -cs.pi/2
x6_f = 0

"""
ocp.set_initial(x1, x1_0)
ocp.set_initial(x2, x2_0)
ocp.set_initial(x3, x3_0)
ocp.set_initial(x4, x4_0)
ocp.set_initial(x5, x5_0)
ocp.set_initial(x6, x6_0)
"""

# Solve
try:
    sol = ocp.solve()
except Exception as e:
    print(str(e))
    sol = ocp.non_converged_solution

#%%
# Post-processing
# ---------------

from pylab import *

# Sample a state/control or expression thereof on a grid
tsa, x1a = sol.sample(x1, grid='control')
tsa, x2a = sol.sample(x2, grid='control')

figure(figsize=(10, 4))
subplot(1, 2, 1)
plot(tsa, x1a, 'o--')
xlabel("Times [s]", fontsize=14)
grid(True)
title('State x1')

subplot(1, 2, 2)
plot(tsa, x2a, 'o--')
legend(['grid_control'])
xlabel("Times [s]", fontsize=14)
title('State x2')
grid(True)

# sphinx_gallery_thumbnail_number = 2

# Refine the grid for a more detailed plot
tsol, usol = sol.sample(u, grid='control')

figure()
step(tsol,usol,where='post')
title("Control signal")
xlabel("Times [s]")
grid(True)

show(block=True)

