#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

"""
Model Predictive Control example
================================

"""

from rockit import *
from casadi import *

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Problem parameters
# -------------------------------
mcart = 0.5                 # cart mass [kg]
m     = 1                   # pendulum mass [kg]
L     = 2                   # pendulum length [m]
g     = 9.81                # gravitation [m/s^2]

nx    = 4                   # the system is composed of 4 states
nu    = 1                   # the system has 1 input
Tf    = 2.0                 # control horizon [s]
Nhor  = 50                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X = vertcat(0.5,0,0,0)  # initial state
final_X   = vertcat(0,0,0,0)    # desired terminal state

Nsim  = 200                 # how much samples to simulate
add_noise = True            # enable/disable the measurement noise addition in simulation
add_disturbance = True      # enable/disable the disturbance addition in simulation

# -------------------------------
# Logging variables
# -------------------------------
pos_history     = np.zeros(Nsim+1)
theta_history   = np.zeros(Nsim+1)
F_history       = np.zeros(Nsim)

# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)

# Define states
pos    = ocp.state()  # [m]
theta  = ocp.state()  # [rad]
dpos   = ocp.state()  # [m/s]
dtheta = ocp.state()  # [rad/s]

# Defince controls
F = ocp.control(nu, order=0)

# Define parameter
X_0 = ocp.parameter(nx);

# Specify ODE
ocp.set_der(pos, dpos)
ocp.set_der(theta, dtheta)
ocp.set_der(dpos, (-m*L*sin(theta)*dtheta*dtheta + m*g*cos(theta)*sin(theta)+F)/(mcart + m - m*cos(theta)*cos(theta)) )
ocp.set_der(dtheta, (-m*L*cos(theta)*sin(theta)*dtheta*dtheta + F*cos(theta)+(mcart+m)*g*sin(theta))/(L*(mcart + m - m*cos(theta)*cos(theta))))

# Lagrange objective
ocp.add_objective(ocp.integral(F*2 + 100*pos**2))

# Initial constraints
X = vertcat(pos,theta,dpos,dtheta)
ocp.subject_to(ocp.at_t0(X)==X_0)
ocp.subject_to(ocp.at_tf(X)==final_X)

# Path constraints
ocp.subject_to(-2 <= (F <= 2 ))
# Note: pos is already fixed at t0 by initial constraints
#       include_first False avoids adding a redundant constraint 
ocp.subject_to(-2 <= (pos <= 2), include_first=False)


# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
ocp.set_value(X_0, current_X)
# Solve
sol = ocp.solve()

# Get discretisd dynamics as CasADi function
Sim_pendulum_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
pos_history[0]   = current_X[0]
theta_history[0] = current_X[1]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

DM.rng(0)

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    tsa, Fsol = sol.sample(F, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_pendulum_dyn(x0=current_X, u=Fsol[0], T=dt)["xf"]
    # Add disturbance at t = 2*Tf
    if add_disturbance:
        if i == round(2*Nhor)-1:
            disturbance = vertcat(0,0,-1e-1,0)
            current_X = current_X + disturbance
    # Add measurement noise
    if add_noise:
        meas_noise = 5e-4*(DM.rand(nx,1)-vertcat(1,1,1,1)) # 4x1 vector with values in [-1e-3, 1e-3]
        current_X = current_X + meas_noise
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X[:4])
    # Solve the optimization problem
    sol = ocp.solve()

    # NOTE: sol.sample/ocp.set_value/ocp.solve
    #       bring about some processing overhead
    #       which may be larger than the actual online optimization
    #       See repeated_solve.py

    # Log data for post-processing
    pos_history[i+1]   = current_X[0].full()
    theta_history[i+1] = current_X[1].full()
    F_history[i]       = Fsol[0]

print("Plot the results")

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)

fig, ax1 = plt.subplots()
ax1.plot(time_sim, pos_history, 'r-')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Cart position [m]', color='r')
ax1.tick_params('y', colors='r')
ax2 = ax1.twinx()
ax2.plot(time_sim, theta_history, 'b-')
ax2.set_ylabel('Pendulum angle [rad]', color='b')
ax2.tick_params('y', colors='b')
ax2.axvline(x=2*Tf, color='k', linestyle='--')
ax2.text(2*Tf+0.1,0.025,'disturbance applied',rotation=90)
fig.tight_layout()

print("Animate results")

# -------------------------------
# Animate results
# -------------------------------
#plt.ion() # Enable plt interactive
if False:# plt.isinteractive():
  fig2, ax3 = plt.subplots(1, 1)
  plt.ion()
  ax3.set_xlabel("X [m]")
  ax3.set_ylabel("Y [m]")
  for k in range(Nsim+1):
      cart_pos_k      = pos_history[k]
      theta_k         = theta_history[k]
      pendulum_pos_k  = vertcat(horzcat(cart_pos_k,0), vertcat(cart_pos_k-L*sin(theta_k),L*cos(theta_k)).T)
      color_k     = 3*[0.95*(1-float(k)/Nsim)]
      ax3.plot(pendulum_pos_k[0,0], pendulum_pos_k[0,1], "s", markersize = 15, color = color_k)
      ax3.plot(pendulum_pos_k[:,0], pendulum_pos_k[:,1], "-", linewidth = 1.5, color = color_k)
      ax3.plot(pendulum_pos_k[1,0], pendulum_pos_k[1,1], "o", markersize = 10, color = color_k)
      plt.pause(dt)

print("show")
plt.show(block=True)
print("end")