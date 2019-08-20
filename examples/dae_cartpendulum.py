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
DAE cart-pendulum example
=========================

Taken from the CasADi example repo
"""

from rockit import *

xref = 0.1 # chariot reference

l = 1. #- -> crane, + -> pendulum
m = 1.
M = 1.
g = 9.81

ocp = Ocp(T=5)

x = ocp.state()
y = ocp.state()
w = ocp.state()
dx = ocp.state()
dy = ocp.state()
dw = ocp.state()

xa = ocp.algebraic()
u = ocp.control()


ocp.set_der(x, dx)
ocp.set_der(y, dy)
ocp.set_der(w, dw)
ddx = (w-x)*xa/m
ddy = g-y*xa/m
ddw = ((x-w)*xa - u)/M
ocp.set_der(dx, ddx)
ocp.set_der(dy, ddy)
ocp.set_der(dw, ddw)
ocp.add_alg((x-w)*(ddx - ddw) + y*ddy + dy*dy + (dx-dw)**2)

ocp.add_objective(ocp.at_tf((x-xref)*(x-xref) + (w-xref)*(w-xref) + dx*dx + dy*dy))
ocp.add_objective(ocp.integral((x-xref)*(x-xref) + (w-xref)*(w-xref)))

ocp.subject_to(-2 <= (u <= 2))

ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(ocp.at_t0(y)==l)
ocp.subject_to(ocp.at_t0(w)==0)
ocp.subject_to(ocp.at_t0(dx)==0)
ocp.subject_to(ocp.at_t0(dy)==0)
ocp.subject_to(ocp.at_t0(dw)==0)
#ocp.subject_to(xa>=0,grid='integrator_roots')

ocp.set_initial(y, l)
ocp.set_initial(xa, 9.81)

# Pick an NLP solver backend
# NOTE: default scaling strategy of MUMPS leads to a singular matrix error 
ocp.solver('ipopt',{"ipopt.linear_solver": "mumps","ipopt.mumps_scaling":0,"ipopt.tol":1e-12} )

# Pick a solution method
method = DirectCollocation(N=50)
ocp.method(method)

# Solve
sol = ocp.solve()

from pylab import *

for e in [x,y,w]:
    ts, xs = sol.sample(e, grid='integrator',refine=100)
    plot(ts, xs)

figure()
plot(*sol.sample(u, grid='integrator',refine=100))

plot(*sol.sample(xa, grid='integrator',refine=100))


show(block=True)
