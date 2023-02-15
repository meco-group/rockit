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


# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
ocp.set_value(X_0, current_X)

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

sol = ocp.solve()

############# Define API for cartpole ################
# Start Query #

cegar_api = ocp.view_api('cegar')

# Note: rockit by itself does not keep names of statescontrols
# If you call view_api on an impact instance you will get proper names
print(cegar_api.get_variable_names())


# Note: this is really parameters, not constants
# constants are inlined in rockit with names dropped
# If you want contants, promote the to parameters 
print(cegar_api.get_constants())

print(cegar_api.get_ODE())


print(cegar_api.get_cost())

print(cegar_api.get_path_constraints())

print(cegar_api.get_initial_constraints())

print(cegar_api.get_sample_time())


"""
# Query ODEs
ocp.get_ODE()
return "posD' == dposD,thetaD == dthetaD, dposD' == (-1.0 * 2.0 * sin(thetaD) * dthetaD * dthetaD + 1.0 * 9.81 * cos(thetaD) * sin(thetaD) + FD) / (0.5 + 1.0 - 1.0 * cos(thetaD) * cos(thetaD)),
dthetaD' == (-1.0 * 2.0 * cos(thetaD) * sin(thetaD) * dthetaD * dthetaD + FD * cos(thetaD) + (0.5 + 1.0) * 9.81 * sin(thetaD)) / (2.0 * (0.5 + 1.0 - 1.0 * cos(thetaD) * cos(thetaD)))"

# Query cost function
ocp.get_cost()
return "costD' == 2.0 * FD + 100.0 * posD * posD"

# Query path constraints
ocp.get_path_constraints()
return "-2 <= (F <= 2)"

# Query initial constraints
ocp.get_initial_constraints()
return "pos=0.5,theta=0,dpos=0,dtheta=0"

# Query sample time
# sample_time = control_horizon/number of control intervals
ocp.get_sample_time()
return "sample_time = 0.04"

# End Query #
"""

