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
import casadi as ca

ocp = Ocp(t0=0, T=10)

# Define two scalar states (vectors and matrices also supported)
p = ocp.state(2)
v = ocp.state(2)
a = ocp.control(2)

ocp.set_der(p, v)

#c = ocp.control(order=5)
#c = ocp.control(order=3)

ocp.set_der(v, a)

# Lagrange objective term: signals in an integrand
ocp.add_objective(ocp.sum(ca.sumsqr(a)))
# Mayer objective term: signals evaluated at t_f = t0_+T
#ocp.add_objective(ocp.at_tf(a**2))

# Path constraints
#  (must be valid on the whole time domain running from `t0` to `tf`,
#   grid options available such as `grid='inf'`)
ocp.subject_to(-1 <= (v <= 1 ),grid='inf')
ocp.subject_to(-10 <= (a <= 10 ),grid='inf')


# Stay outside an obstacle
c = ca.vertcat(1,2)
#ocp.subject_to( ca.norm_2(p-c) >= 10,refine=100) #,group_refine=LseGroup(margin_abs=5))

# Boundary constraints
ocp.subject_to(ocp.at_t0(p) == 0)
ocp.subject_to(ocp.at_tf(p) == 1)

ocp.subject_to(ocp.at_t0(v) == 0)
ocp.subject_to(ocp.at_tf(v) == 0)

#%%
# Solving the problem
# -------------------

# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt')

# Pick a solution method
#  e.g. SingleShooting, MultipleShooting, DirectCollocation
#
#  N -- number of control intervals
#  M -- number of integration steps per control interval
#  grid -- could specify e.g. UniformGrid() or GeometricGrid(4)
method = SplineMethod(N=10)
#method = MultipleShooting(N=10)
ocp.method(method)

# Solve
sol = ocp.solve()

# In case the solver fails, you can still look at the solution:
# (you may need to wrap the solve line in try/except to avoid the script aborting)
sol = ocp.non_converged_solution

#%%
# Post-processing
# ---------------

from pylab import *

# Sample a state/control or expression thereof on a grid
tsol, psol = sol.sample(p, grid='control')
tsol_f, psol_f = sol.sample(p, grid='control',refine=100)

figure(figsize=(10, 4))
plot(tsol, psol, 'b.')
plot(tsol_f, psol_f,'b')
title("p")

tsol, vsol = sol.sample(v, grid='control')
tsol_f, vsol_f = sol.sample(v, grid='control',refine=100)

figure(figsize=(10, 4))
plot(tsol, vsol, 'b.')
plot(tsol_f, vsol_f,'b')
title("v")

tsol, asol = sol.sample(a, grid='control')
tsol_f, asol_f = sol.sample(a, grid='control',refine=100)

figure(figsize=(10, 4))
plot(tsol, asol, 'b.')
plot(tsol_f, asol_f,'b')
title("a")



show(block=True)
