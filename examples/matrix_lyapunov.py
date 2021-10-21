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
Example with matrix state
=========================

The example uses the Lyapunov differential equation to approximate
state covariance along the trajectory
"""

# time optimal example for mass-spring-damper system
from rockit import *
from casadi import sumsqr, vertcat, sin, cos, vec, diag, horzcat, sqrt, jacobian
import numpy as np

ocp = Ocp(T=1.0)

x = ocp.state(2) # two states
u = ocp.control()

# nominal dynamics
der_state = vertcat(x[1],-0.1*(1-x[0]**2)*x[1] - x[0] + u)
ocp.set_der(x, der_state)

# Lyapunov state
P = ocp.state(2, 2)

# Lyapunov dynamics
A = jacobian(der_state, x)
ocp.set_der(P, A @ P + P @ A.T)

ocp.subject_to(ocp.at_t0(x) == vertcat(0.5,0))

P0 = diag([0.01**2,0.1**2])
ocp.subject_to(ocp.at_t0(P) == P0)
ocp.set_initial(P, P0)

ocp.subject_to(-40 <= (u <= 40))

ocp.subject_to(x[0] >= -0.25)

# 1-sigma bound on x1
sigma = sqrt(horzcat(1,0) @ P @ vertcat(1,0))

# time-dependent bound 
def bound(t):
  return 2 + 0.1*cos(10*t)

# Robustified path constraint
ocp.subject_to(x[0] <= bound(ocp.t) - sigma)

# Tracking objective
ocp.add_objective(ocp.integral(sumsqr(x[0]-3)))

ocp.solver('ipopt')

ocp.method(MultipleShooting(N=40, M=6, intg='rk'))

sol = ocp.solve()


# Post-processing

import matplotlib.pyplot as plt

ts, xsol = sol.sample(x[0], grid='control')

plt.plot(ts, xsol, '-o')
plt.plot(ts, bound(ts))

plt.legend(["x1"])
ts, Psol = sol.sample(P,grid = 'control')

o = np.array([[1,0]])

sigma = []
for i in range(len(ts)):
  sigma.append(float(np.sqrt(o @ Psol[i,:,:] @ o.T)))
sigma =  np.array(sigma)
plt.plot([ts,ts],[xsol-sigma,xsol+sigma],'k')

plt.legend(('OCP trajectory x1','bound on x1'))
plt.xlabel('Time [s]')
plt.ylabel('x1')
plt.show()
