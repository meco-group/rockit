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
A Hello World Example
===================

Some basic example on solving an Optimal Control Problem with rockit.
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

# Define one piecewise constant control input
#  (use `order=1` for piecewise linear)
u = ocp.control()

# Compose time-dependent expressions a.k.a. signals
#  (explicit time-dependence is supported with `ocp.t`)
e = 1 - x2**2
#e = 1
# Specify differential equations for states
#  (DAEs also supported with `ocp.algebraic` and `add_alg`)
ocp.set_der(x1, e * x1 - x2 + u)
ocp.set_der(x2, x1)

# Lagrange objective term: signals in an integrand
ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))
# Mayer objective term: signals evaluated at t_f = t0_+T
ocp.add_objective(ocp.at_tf(x1**2))

# Path constraints
#  (must be valid on the whole time domain running from `t0` to `tf`,
#   grid options available such as `grid='inf'`)
ocp.subject_to(x1 >= -0.25)
ocp.subject_to(-1 <= (u <= 1 ))

# Boundary constraints
ocp.subject_to(ocp.at_t0(x1) == 0)
ocp.subject_to(ocp.at_t0(x2) == 1)

#%%
# Solving the problem
# -------------------

# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt')

# Pick a solution method
method = external_method('grampc',N=20)
#method = MultipleShooting(N=20)
ocp.method(method)
#ocp.method(MultipleShooting(N=50))

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