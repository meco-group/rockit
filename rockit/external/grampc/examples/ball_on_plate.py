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
Ball on plate 
===================


A line, really
Computational complexity certifcation of gradient methods
for real-time model predictive control. Ph.D. thesis, ETH Zürich (2012).
Stefan Richter
 
"""


# Make available sin, cos, etc
from numpy import *
# Import the project
from rockit import *
import casadi as cs
#%%
# Problem specification
# ---------------------

ocp = Ocp(T=0.3)


u = ocp.control()
x = ocp.state(2)

wx = cs.vertcat(100, 10)
wu = 180
wTx = cs.vertcat( 100, 10)

pCost = cs.vertcat(100, 10, 180, 100, 10, -0.2, 0.2, -0.1, 0.1)

xdes = ocp.parameter(2)
udes = ocp.parameter()

ocp.set_der(x, cs.vertcat(-0.04*u+x[1],-7.01*u))

Wx = cs.diag(wx)
Wu = cs.diag(wu)
WTx = cs.diag(wTx)

#quad = lambda W, b : cs.bilin(W,b,b)
quad = lambda w, b : cs.dot(w,b**2)

ocp.add_objective(ocp.integral(1/2*quad(wx,x-xdes)))
ocp.add_objective(ocp.integral(1/2*quad(wu,u-udes)))
ocp.add_objective(ocp.at_tf(1/2*quad(wTx,x-xdes)))

ocp.set_value(xdes, cs.vertcat(-0.2,0.0))
ocp.set_value(udes, 0.0)

ocp.subject_to(ocp.at_t0(x)==cs.vertcat(0.1, 0.01))
ocp.set_initial(u, 0.0)


ocp.subject_to(-0.0524 <= (u <= 0.0524))

xlim = cs.vertcat(0.2, 0.1)
ocp.subject_to(-xlim <= (x <= xlim))
#ocp.subject_to(-xlim[0]-x[0]<=0)
#ocp.subject_to(-xlim[0]+x[0]<=0)
#ocp.subject_to(-xlim[1]-x[1]<=0)
#ocp.subject_to(-xlim[1]+x[1]<=0)

# Solving the problem
# -------------------

# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt')
# Pick a solution method

grampc_options={}
grampc_options["MaxMultIter"] = 3
grampc_options["AugLagUpdateGradientRelTol"] = 1e0
grampc_options["ConstraintsAbsTol"] = 1e-3


method = external_method('grampc',N=50,expand=True,grampc_options=grampc_options)
ocp.method(method)
#ocp.method(MultipleShooting(N=50))


# Solve
sol = ocp.solve()

t, usol = sol.sample(u,grid='control')

t, xsol = sol.sample(x,grid='control')

import pylab as plt

plt.figure()
plt.plot(t,xsol)
plt.figure()
plt.plot(t,usol)
plt.show()
