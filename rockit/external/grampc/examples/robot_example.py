"""
A rockit implemantation of the robot problem
"""
import rockit as roc
import numpy as np
import casadi as cs

# Start an optimal control environment with a time horizon of 10 seconds
# starting from t0=0s.
#  (free-time problems can be configured with `FreeTime(initial_guess)`)
ocp = roc.Ocp(t0=0, T=10)

# Define two scalar states (vectors and matrices also supported)
x1 = ocp.state()
x2 = ocp.state()
x3 = ocp.state()
x4 = ocp.state()
x5 = ocp.state()
x6 = ocp.state()

# Define one piecewise constant control input
#  (use `order=1` for piecewise linear)
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

x0 = cs.vertcat(cs.pi/2, -cs.pi/2, 0, -cs.pi/2, cs.pi/2, 0)
xf = cs.vertcat(-cs.pi/2, cs.pi/2, 0, cs.pi/2, -cs.pi/2, 0)
# Compose time-dependent expressions a.k.a. signals
#  (explicit time-dependence is supported with `ocp.t`)
# e = 1 - x2**2

# Specify differential equations for states
#  (DAEs also supported with `ocp.algebraic` and `add_alg`)
ocp.set_der(x1, u1)
ocp.set_der(x2, u2)
ocp.set_der(x3, u3)
ocp.set_der(x4, u4)
ocp.set_der(x5, u5)
ocp.set_der(x6, u6)

# Lagrange objective term: signals in an integrand
# ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))
ocp.add_objective(ocp.integral(0.5 * u.T @ u))
# Mayer objective term: signals evaluated at t_f = t0_+T
# ocp.add_objective(ocp.at_tf(x1**2))
offset_left = cs.vertcat(0, 0, 0)
offset_right = cs.vertcat(1, 0, cs.pi)

# p_L = cs.vertcat(a1*cs.cos(x1) + a2*cs.cos(x1+x2) + a3*cs.cos(x1 + x2 + x3) + offset_left[0],
#                  a1*cs.sin(x1) + a2*cs.sin(x1+x2) + a3*cs.sin(x1 + x2 + x3) + offset_left[1],
#                  x1 + x2 + x3 + offset_left[2])

p_L = cs.vertcat(a1*cs.cos(x1) + a2*cs.cos(x1+x2) + a3*cs.cos(x1 + x2 + x3),
                 a1*cs.sin(x1) + a2*cs.sin(x1+x2) + a3*cs.sin(x1 + x2 + x3))

# p_R = cs.vertcat(1 + a4*cs.cos(x4) + a5*cs.cos(x4+x5) + a6*cs.cos(x4 + x5 + x6),
#                  a4*cs.sin(x4) + a5*cs.sin(x4+x5) + a6*cs.sin(x4 + x5 + x6),
#                  x4 + x5 + x6 + offset_right[2])

p_R = cs.vertcat(1 - a4*cs.cos(x4) - a5*cs.cos(x4+x5) - a6*cs.cos(x4 + x5 + x6),
                 -a4*cs.sin(x4) -a5*cs.sin(x4+x5) -a6*cs.sin(x4 + x5 + x6))
# p_R = cs.vertcat(1 + a4*cs.cos(x4) + a5*cs.cos(x4+x5) + a6*cs.cos(x4 + x5 + x6),
#                  a4*cs.sin(x4) + a5*cs.sin(x4+x5) + a6*cs.sin(x4 + x5 + x6))
# p_R = cs.vertcat(1 + a4*cs.cos(-x4) + a5*cs.cos(-x4-x5) + a6*cs.cos(-x4  -x5 -x6),
#                  a4*cs.sin(-x4) + a5*cs.sin(-x4-x5) + a6*cs.sin(-x4 -x5 -x6))

g = p_L - p_R #- cs.vertcat(0, 0, cs.pi)

u_max = cs.vertcat(1, 1, 1, 1, 1, 1)

# Path constraints
#  (must be valid on the whole time domain running from `t0` to `tf`,
#   grid options available such as `grid='inf'`)
ocp.subject_to(g == 0, include_first=False, include_last=False)
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

# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt', {"error_on_fail":False, 'ipopt':{"max_iter": 1000, 'hessian_approximation':'exact', 'limited_memory_max_history' : 5, 'print_level':5}})

# Pick a solution method
N=20
method = roc.MultipleShooting(N=N, intg='rk')
# method = external_method('grampc')
ocp.method(method)

# Set initial guesses for states, controls and variables.
#  Default: zero
# ocp.set_initial(x2, 0)                 # Constant
# ocp.set_initial(x1, ocp.t/10)          # Function of time
# ocp.set_initial(u, np.linspace(0, 0.1, 6)) # Matrix
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

ocp.set_initial(x1, x1_0)
ocp.set_initial(x2, x2_0)
ocp.set_initial(x3, x3_0)
ocp.set_initial(x4, x4_0)
ocp.set_initial(x5, x5_0)
ocp.set_initial(x6, x6_0)

# ocp.set_initial(x1, x1_0*(1-dist)+dist*x1_f)
# ocp.set_initial(x2, x2_0*(1-dist)+dist*x2_f)
# ocp.set_initial(x3, x3_0*(1-dist)+dist*x3_f)
# ocp.set_initial(x4, x4_0*(1-dist)+dist*x4_f)
# ocp.set_initial(x5, x5_0*(1-dist)+dist*x5_f)
# ocp.set_initial(x6, x6_0*(1-dist)+dist*x6_f)

# Solve
sol = ocp.solve()

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

