from ocpx import *
from casadi import *
from casadi.tools import *

import matplotlib.pyplot as plt

# -------------------------------
# Problem parameters
# -------------------------------
M = 1.0
m = 0.1
l = 0.8
g = 9.81
nx = 4
nu = 1
Tf = 1
Nhor = 20
dt = Tf/Nhor
current_X = [0, 0, pi, 0]

# -------------------------------
# Continuous system dynamics
# -------------------------------
def pendulum_dynamics(x,u):
    dpos = x[1]
    ddpos = (-m*l*sin(x[2])*x[3]*x[3] + m*g*cos(x[2])*sin(x[2])+u)/(M + m - m*cos(x[2])*cos(x[2]))
    dtheta = x[3]
    ddtheta = (-m*l*cos(x[2])*sin(x[2])*x[3]*x[3] + u*cos(x[2])+(M+m)*g*sin(x[2]))/(l*(M + m - m*cos(x[2])*cos(x[2])))
    return vertcat(dpos, ddpos, dtheta, ddtheta)

# -------------------------------
# Discretized system dynamics
# -------------------------------
def rk4_disc(f):
    x = MX.sym('x',nx)
    u = MX.sym('u',nu)
    dt = MX.sym('dt')
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return Function('F', [x, u, dt], [x + dt/6*(k1 + 2*k2 + 2*k3 + k4)], ['x0', 'u', 'DT'], ['xf'])

Sim_pendulum_dyn = rk4_disc(pendulum_dynamics)

# -------------------------------
# Set OCP
# -------------------------------
ocp = OcpMultiStage()

stage = ocp.stage(t0=0,T=Tf)

# Define states
X = stage.state(nx)

# Defince controls
F = stage.control(nu,order=1)

# Define parameter
X_0 = stage.parameter(nx);


# X[0] <-- pos
# X[1] <-- dpos
# X[2] <-- theta
# X[3] <-- dtheta

# Specify ODE
stage.set_der(X, pendulum_dynamics(X,F))

# Lagrange objective
stage.add_objective(stage.integral(1e1*X[0]**2 + 1e-3*X[1]**2 + 1e-1*X[2]**2 + 1e-3*X[3]**2 + 1e-3*F**2)) # 1e1*X[0]**2 + 1e-3*X[1]**2 + 1e-1*X[2]**2 + 1e-3*X[3]**2

# Path constraints
stage.subject_to(F<=40)
stage.subject_to(F>=-40)
stage.subject_to(X[0]>=-0.45)
stage.subject_to(X[0]<=0.45)

# Initial constraints
stage.subject_to(stage.at_t0(X)==X_0)
# stage.subject_to(sumsqr(stage.at_tf(X))<=1e-2)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=Nhor,M=1,intg='rk'))


# -------------------------------
# Solve the OCP wrt a parameter value
# -------------------------------
# Set initial value for parameters
stage.set_value(X_0, current_X)
# Solve
sol = ocp.solve()

# First solution
tsa,Fsol = sol.sample(stage,F,grid=stage.grid_control)
tsb,thetasol = sol.sample(stage,X[2],grid=stage.grid_control)

#
#

print(current_X)

current_X = Sim_pendulum_dyn(current_X, Fsol[0], dt)

print(current_X)


# # Plot the result
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(tsa,Fsol,'.-')
ax[1].plot(tsb,thetasol,'.-')
for i in range(2):
    ax[i].set_xlabel('Time [s]', fontsize=14)
ax[0].set_ylabel('F [N]', fontsize=14)
ax[1].set_ylabel('Theta [rad]', fontsize=14)
plt.show(block=True)
