from ocpx import *
from casadi import *
from casadi.tools import *

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Problem parameters
# -------------------------------
mcart = 0.5                 # cart mass [kg]
m = 1                       # pendulum mass [kg]
L = 2                       # pendulum length [m]
g = 9.81                    # gravitation [m/s^2]

nx = 4                      # the system is composed of 4 states
nu = 1                      # the system has 1 input
Tf = 2                      # control horizon [s]
Nhor = 100                  # number of control intervals
dt = Tf / Nhor                # sample time

current_X = vertcat(0.5, 0, 0, 0)      # initial state
final_X = vertcat(0, 0, 0, 0)        # desired terminal state

Nsim = 10                  # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
pos_history = np.zeros(Nsim + 1)
theta_history = np.zeros(Nsim + 1)
F_history = np.zeros(Nsim)

# -------------------------------
# Continuous system dynamics
# -------------------------------


def pendulum_dynamics(x, u):
    # states: pos [m], theta [rad], dpos [m/s], dtheta [rad/s]
    dpos = x[2]
    dtheta = x[3]
    ddpos = (-m * L * sin(x[1]) * x[3] * x[3] + m * g * cos(x[1]) *
             sin(x[1]) + u) / (mcart + m - m * cos(x[1]) * cos(x[1]))
    ddtheta = (-m * L * cos(x[1]) * sin(x[1]) * x[3] * x[3] + u * cos(x[1]) + (
        mcart + m) * g * sin(x[1])) / (L * (mcart + m - m * cos(x[1]) * cos(x[1])))
    return vertcat(dpos, dtheta, ddpos, ddtheta)

# -------------------------------
# Discretized system dynamics (only used for dynamics simulation in the MPC loop)
# -------------------------------


def rk4_disc(f):
    x = MX.sym('x', nx)
    u = MX.sym('u', nu)
    dt = MX.sym('dt')
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return Function('F', [x, u, dt], [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)], ['x0', 'u', 'DT'], ['xf'])


Sim_pendulum_dyn = rk4_disc(pendulum_dynamics)

# -------------------------------
# Set OCP
# -------------------------------
ocp = OcpMultiStage()

stage = ocp.stage(t0=0, T=Tf)

# Define states
X = stage.state(nx)

# Defince controls
F = stage.control(nu, order=0)

# Define parameter
X_0 = stage.parameter(nx)

# Specify ODE
stage.set_der(X, pendulum_dynamics(X, F))

# Lagrange objective
stage.add_objective(stage.integral(sumsqr(F) + 100 * sumsqr(X[0])))

# Path constraints
stage.subject_to(F <= 2)
stage.subject_to(F >= -2)
stage.subject_to(-2 <= X[0])
stage.subject_to(X[0] <= 2)

# Initial constraints
stage.subject_to(stage.at_t0(X) == X_0)
stage.subject_to(stage.at_tf(X) == final_X)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=Nhor, M=1, intg='rk'))

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
stage.set_value(X_0, current_X)
# Solve
sol = ocp.solve()

# # Plot the results from offline solution
tsa, thetasol = sol.sample(stage, X[1], grid='control')
tsb, possol = sol.sample(stage, X[0], grid='control')
tsc, Fsol = sol.sample(stage, F, grid='control')

fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].plot(tsa, thetasol, '.-')
ax[1].plot(tsb, possol, '.-')
ax[2].plot(tsc, Fsol, '.-')
for i in range(2):
    ax[i].set_xlabel('Time [s]', fontsize=14)
ax[0].set_ylabel('Theta [rad]', fontsize=14)
ax[1].set_ylabel('Pos [m]', fontsize=14)
ax[2].set_ylabel('Force [N]', fontsize=14)
plt.show(block=True)

# # Log data for post-processing
# pos_history[0] = current_X[0]
# theta_history[0] = current_X[1]
#
# # -------------------------------
# # Simulate the MPC solving the OCP (with the updated state) several times
# # -------------------------------
# for i in range(Nsim):
#     # Get the solution from sol
#     tsa,Fsol = sol.sample(stage,F,grid='control')
#     # Simulate dynamics (applying the first control input) and update the current state
#     current_X = Sim_pendulum_dyn(current_X, Fsol[0], dt)
#     # Set the parameter X0 to the new current_X
#     stage.set_value(X_0, current_X)
#     # Solve the optimization problem
#     sol = ocp.solve()
#
#     # Log data for post-processing
#     pos_history[i+1] = current_X[0].full()
#     theta_history[i+1] = current_X[1].full()
#     F_history[i] = Fsol[0]
#
#
#
# print(pos_history)
# print(theta_history)
# print(F_history)
#
# # # # Plot the result
# # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# # ax[0].plot(tsa,Fsol,'.-')
# # ax[1].plot(tsb,thetasol,'.-')
# # for i in range(2):
# #     ax[i].set_xlabel('Time [s]', fontsize=14)
# # ax[0].set_ylabel('F [N]', fontsize=14)
# # ax[1].set_ylabel('Theta [rad]', fontsize=14)
# # plt.show(block=True)
