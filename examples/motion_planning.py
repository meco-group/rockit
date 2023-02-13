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
Motion planning
===============

Simple motion planning with circular obstacle
"""

from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr

ocp = Ocp(T=FreeTime(10.0))

# Bicycle model
# (S. LaValle. Planning Algorithms. Cambridge University Press, 2006, pp. 724–725.)

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

delta = ocp.control()
V     = ocp.control()

L = 1

ocp.set_der(x, V*cos(theta))
ocp.set_der(y, V*sin(theta))
ocp.set_der(theta, V/L*tan(delta))

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(ocp.at_t0(y)==0)
ocp.subject_to(ocp.at_t0(theta)==pi/2)

# Final constraint
ocp.subject_to(ocp.at_tf(x)==0)
ocp.subject_to(ocp.at_tf(y)==10)

ocp.set_initial(x,0)
ocp.set_initial(y,ocp.t)
ocp.set_initial(theta, pi/2)
ocp.set_initial(V,1)

ocp.subject_to(0 <= (V<=1))
ocp.subject_to( -pi/6 <= (delta<= pi/6))

# Round obstacle
p0 = vertcat(0.2,5)
r0 = 1

p = vertcat(x,y)
ocp.subject_to(sumsqr(p-p0)>=r0**2)

# Minimal time
ocp.add_objective(ocp.T)
ocp.add_objective(ocp.integral(x**2))

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=20,M=4,intg='rk'))

# solve
sol = ocp.solve()

from pylab import *
figure()

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')

plot(xs, ys,'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')


ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)

plot(xs, ys, '-')

ts = np.linspace(0,2*pi,1000)
plot(p0[0]+r0*cos(ts),p0[1]+r0*sin(ts),'r-')

axis('equal')
show(block=True)
