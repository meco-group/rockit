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
from ast import Mult
from numpy import *
# Import the project
from rockit import *
#%%
# Problem specification
# ---------------------


T0 = 5.25
time_opt = True
ocp = Ocp(t0=0, T=FreeTime(T0) if time_opt else T0)

# Define two scalar states (vectors and matrices also supported)
x1 = ocp.state()
x2 = ocp.state()

u = ocp.control()

ocp.set_der(x1, x2)
ocp.set_der(x2, u)

ocp.add_objective(ocp.integral(0.1*u**2))
ocp.add_objective(ocp.at_tf(1*ocp.T))

if time_opt:
    ocp.subject_to(1.0 <= (ocp.T <= 10 ))

ocp.subject_to(x2 <= 0.5)
ocp.subject_to(-1 <= (u <= 1 ))

# Boundary constraints
ocp.subject_to(ocp.at_t0(x1) == -1)
ocp.subject_to(ocp.at_t0(x2) == -1)

ocp.subject_to(ocp.at_tf(x1) == 0)
ocp.subject_to(ocp.at_tf(x2) == 0)

#%%
# Solving the problem
# -------------------

# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt')

grampc_options = {}
grampc_options["MaxGradIter"] = 200
grampc_options["MaxMultIter"] = 1
grampc_options["ShiftControl"] = "off"
grampc_options["Integrator"] = "euler"
grampc_options["LineSearchMax"] = 1e2

grampc_options["PenaltyMin"] = 1e1
grampc_options["PenaltyIncreaseFactor"] = 1.25
grampc_options["PenaltyDecreaseFactor"] = 1.0

grampc_options["ConstraintsAbsTol"] = 1e-6
grampc_options["ConvergenceCheck"] = "on"
grampc_options["ConvergenceGradientRelTol"] = 1e-9
if time_opt:
    grampc_options["OptimTimeLineSearchFactor"] = 1.75

# Pick a solution method
method = external_method('grampc',N=40,grampc_options=grampc_options)
#method = MultipleShooting(N=40)
ocp.method(method)

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