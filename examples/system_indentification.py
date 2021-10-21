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
System identification
=====================

Shows an example of sysid/parameter estimation with rockit.
"""


# Make available sin, cos, etc
from numpy import *
# Import the project
from rockit import *

#%%
# Problem specification
# ---------------------

# Start an optimal control environment with a time horizon of 10 seconds
# starting from t0=0s.
#  (free-time problems can be configured with `FreeTime(initial_guess)`)
ocp = Ocp(t0=0, T=10)

# Define two scalar states (vectors and matrices also supported)
x1 = ocp.state()
x2 = ocp.state()

# Measurements of state
x1_meas = [ 2.52469687e-36 ,-2.50000010e-01 ,-2.50000009e-01, -2.50000006e-01,
 -1.48309122e-01, -3.72646090e-02,  6.56395958e-03,  1.06163067e-02,
  5.16887935e-03,  1.71826868e-03, -2.80386510e-04]
x2_meas = [ 1.00000000e+00,  8.68298025e-01,  6.04317331e-01,  3.36762795e-01,
  1.11997054e-01,  3.60719567e-04, -2.04611315e-02, -1.14118570e-02,
 -2.26223705e-03,  1.82764512e-03,  2.79793225e-03]

x1_meas_param = ocp.parameter(grid='control')
x2_meas_param = ocp.parameter(grid='control')
ocp.set_value(x1_meas_param, x1_meas[:-1])
ocp.set_value(x2_meas_param, x2_meas[:-1])

# Measurements of control
u_meas = [ 7.19549789e-01,  8.55058241e-01,  6.76514569e-01  ,5.29132015e-01,
  2.68151632e-01  ,5.06393975e-02 ,-2.12432317e-02 ,-2.09098487e-02,
 -7.43432715e-03, -4.75629541e-04]

# Define one piecewise constant control input
#  (use `order=1` for piecewise linear)
u = ocp.parameter(grid='control')
ocp.set_value(u, u_meas)

# Unknown system parameters
alpha = ocp.variable()
beta = ocp.variable()

# Initial guesses for system parameters
ocp.set_initial(alpha, 2)
ocp.set_initial(beta, 0.5)

# Compose time-dependent expressions a.k.a. signals
e = alpha - beta*x2**2

# Specify differential equations for states
ocp.set_der(x1, e * x1 - x2 + u)
ocp.set_der(x2, x1)

error_sq = (x1-x1_meas_param)**2+(x2-x2_meas_param)**2

# Objective: sum of error_sq
ocp.add_objective(ocp.sum(error_sq))

# Pick an NLP solver backend
ocp.solver('ipopt')

# Pick a solution method
method = MultipleShooting(N=10, intg='rk')
ocp.method(method)

# Exploit the availability of measurements
ocp.set_initial(x1, x1_meas)
ocp.set_initial(x2, x2_meas)

# Solve
sol = ocp.solve()

print("Estimated alpha (true answer: 1) = ")
print(sol.value(alpha))
print("Estimated beta (true answer: 1) = ")
print(sol.value(beta))
