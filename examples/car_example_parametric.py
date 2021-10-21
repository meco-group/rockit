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
Car accelerating on a linear track
====================================


"""

from rockit import *
from numpy import sin, pi
from casadi import vertcat
import numpy as np
import matplotlib.pyplot as plt

ocp = Ocp(T=FreeTime(1.0))

# Define constants
m = 500
c = 2
d = 1000

# Define states
p = ocp.state()
v = ocp.state()

# Defince controls
F = ocp.control()

# Specify ODE
ocp.set_der(p, v)
ocp.set_der(v, 1/m * (F - c * v**2))

# Lagrange objective
ocp.add_objective(ocp.T)

# Define parameters
F_max = ocp.parameter(grid='control')
p0 = ocp.parameter()

# Path constraints
ocp.subject_to(-F_max <= (F<= F_max))
ocp.subject_to(v >= 0)

# Initial constraints
ocp.subject_to(ocp.at_t0(p)==p0)
ocp.subject_to(ocp.at_t0(v)==0)

# End constraints
ocp.subject_to(ocp.at_tf(p)==d)
ocp.subject_to(ocp.at_tf(v)==0)

# Pick a solver
ocp.solver('ipopt')

# Choose a solution method
ocp.method(MultipleShooting(N=20,M=1,intg='rk'))

# Set values for parameters
ocp.set_value(p0, 1)
ocp.set_value(F_max, 2500*np.ones(20))

# Translate problem to function
T = ocp.value(ocp.T)
states = ocp.sample(vertcat(p,v),grid='control')[1]
controls = ocp.sample(F,grid='control-')[1]
test = ocp.to_function('test', [p0, ocp.sample(F_max,grid='control-')[1], T, states, controls], [ocp.sample(F,grid='control')[1], T, states, controls])

# Initial value
sol_T = 1.0
sol_states = 0
sol_controls = 0

# Solve problem for different values for parameters, initializing with previous solution
signal1, sol_T, sol_states1, sol_controls = test(0, 2500*np.ones(20), sol_T, sol_states, sol_controls)
signal2, sol_T, sol_states2, sol_controls = test(0, 2500*np.ones(20), sol_T, sol_states1, sol_controls)
signal3, sol_T, sol_states3, sol_controls = test(0, 2000*np.ones(20), sol_T, sol_states2, sol_controls)


print(signal1[:4], sol_states1[:4])
print(signal2[:4], sol_states2[:4])
print(signal3[:4], sol_states3[:4])
