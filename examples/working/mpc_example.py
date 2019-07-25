from ocpx import *
from casadi import *
from casadi.tools import *

import matplotlib.pyplot as plt

# -------------------------------
# Problem parameters
# -------------------------------
mcart = 0.5                 # cart mass [kg]
m = 1                       # pendulum mass [kg]
L = 0.2                     # pendulum length [m]
g = 9.81                    # gravitation [m/s^2]

nx = 4                      # the system is composed of 4 states
nu = 1                      # the system has 1 input
Tf = 0.5                    # control horizon [s]
Nhor = 50                   # number of control intervals
dt = Tf/Nhor                # sample time

current_X = [0, pi, 0, 0]   # initial state
final_X = [0, 0, 0, 0]      # desired terminal state

Nsim = 10                   # how much samples to simulate

# -------------------------------
# Continuous system dynamics
# -------------------------------
def pendulum_dynamics(x,u):
    # states: pos [m], theta [rad], dpos [m/s], dtheta [rad/s]
    dpos = x[2]
    dtheta = x[3]
    ddpos = (u+m*L*x[3]*x[3]*sin(x[1])-m*g*sin(x[1])*cos(x[1]))/(mcart+m-m*cos(x[1])*cos(x[1]))
    ddtheta = g/L*sin(x[1])-cos(x[1])*ddpos
    return vertcat(dpos, dtheta, ddpos, ddtheta)

# -------------------------------
# Discretized system dynamics (only used for dynamics simulation in the MPC loop)
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
F = stage.control(nu,order=0)

# Define parameter
X_0 = stage.parameter(nx);

# Specify ODE
stage.set_der(X, pendulum_dynamics(X,F))

# Lagrange objective
stage.add_objective(stage.integral(sumsqr(F) + 1e1*sumsqr(X[0]) + 1e2*sumsqr(X[1]) + 1e-1*sumsqr(X[2]) + 1e-1*sumsqr(X[3]))) # 1e1*X[0]**2 + 1e-3*X[1]**2 + 1e-1*X[2]**2 + 1e-3*X[3]**2
stage.add_objective(1e2*sumsqr(stage.at_tf(X)-final_X))

# Path constraints
stage.subject_to(F<=5)
stage.subject_to(F>=-5)
stage.subject_to(X[0]>=-3)
stage.subject_to(X[0]<=3)

# Initial constraints
stage.subject_to(stage.at_t0(X)==X_0)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
stage.set_value(X_0, current_X)
# Solve
sol = ocp.solve()

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------
for x in range(Nsim):
    # Get the solution from sol
    tsa,Fsol = sol.sample(stage,F,grid=stage.grid_control)
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_pendulum_dyn(current_X, Fsol[0], dt)
    # Set the parameter X0 to the new current_X
    stage.set_value(X_0, current_X)
    # Solve the optimization problem
    sol = ocp.solve()

# -------------------------------
# Get the final value of the state vector
# -------------------------------
# Get the solution from sol
tsa,Fsol = sol.sample(stage,F,grid=stage.grid_control)
# Simulate dynamics (applying the first control input) and update the current state
current_X = Sim_pendulum_dyn(current_X, Fsol[0], dt)

print(current_X)
# # # Plot the result
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].plot(tsa,Fsol,'.-')
# ax[1].plot(tsb,thetasol,'.-')
# for i in range(2):
#     ax[i].set_xlabel('Time [s]', fontsize=14)
# ax[0].set_ylabel('F [N]', fontsize=14)
# ax[1].set_ylabel('Theta [rad]', fontsize=14)
# plt.show(block=True)
