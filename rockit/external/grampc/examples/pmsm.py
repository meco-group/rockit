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
permanent magnet synchronous machine
===================


See manual of Grampc

The solution is to high precision equal to the PMSM example in grampc repo
However, very different from MultipoleShooting solution
 
"""


# Make available sin, cos, etc
from numpy import *
# Import the project
from rockit import *
import casadi as cs
#%%
# Problem specification
# ---------------------

ocp = Ocp(T=5e-3)

Ld = Lq = 17.5e-3 # Inductivities [H]
R = 3.5 # Stator resistance [Ω]
ψp = 0.17 # permanent magnet flux [V s]
zp = 3 # Numbe rof pole pairs
J = 0.9e-3 # Moment of inertia [kg m^2]
μf = 0.4e-3 # Friction coefficient [N m s]
TL = 0

id = ocp.state() # Current
iq = ocp.state()
ω = ocp.state() # Electrical rotor speed
ϕ = ocp.state() # Electrical angle

ud = ocp.control()
uq = ocp.control()
u = cs.vertcat(ud,uq)

ocp.set_der(id, (-R*id+Lq*ω*iq+ud)/Ld)
ocp.set_der(iq, (-R*iq-(ψp+Ld*id)*ω+uq)/Lq)
ocp.set_der(ω, (0.5*(-2*zp*TL+3*zp**2*(ψp+(Ld-Lq)*id)*iq-2*μf*ω)/J))
ocp.set_der(ϕ, ω)

Imax = 10              # Maximum current [A]
Umax = sqrt(560**2/3) # Maximum voltage [V]

#ocp.subject_to((ud**2+uq**2)/Umax**2<=Umax**2/Umax**2)
#ocp.subject_to((id**2+iq**2)/Imax**2<=Imax**2/Imax**2)
ocp.subject_to((ud**2+uq**2-Umax**2)/Umax**2<=0)
ocp.subject_to((id**2+iq**2-Imax**2)/Imax**2<=0)

ocp.subject_to(-Umax<= (u <=Umax))

ocp.subject_to(ocp.at_t0(ocp.x)==cs.vertcat(0.01,0.03,0.07,0.11))

q1 = 8  # [A^-2]
q2 = 200 # [A^-2]
R = cs.diag(cs.vertcat(0.001,0.001)) # [V^-2]

id_des = 0  # [A]
iq_des = 10 # [A]
u_des = cs.vertcat(0, 0) # [V]


quad = lambda W, b : cs.bilin(W,b,b)

ocp.add_objective(ocp.integral( q1*(id-id_des)**2 ))
ocp.add_objective(ocp.integral( q2*(iq-iq_des)**2 ))
ocp.add_objective(ocp.integral( quad(R, u-u_des)   ))

grampc_options={}
grampc_options["MaxGradIter"] = 3000
grampc_options["MaxMultIter"] = 3000
grampc_options["ConstraintsAbsTol"] = 1e-3
grampc_options["ConvergenceCheck"] = "on"

method = external_method('grampc',N=100,expand=True,grampc_options=grampc_options)

#ocp.solver("ipopt")
#ocp.method(DirectCollocation(N=100))

ocp.method(method)

# Solve
sol = ocp.solve()

t, usol = sol.sample(u,grid='control')

t, xsol = sol.sample(ocp.x,grid='control')

#print("obj=",sol.value(ocp.objective))

import pylab as plt

plt.figure()
plt.plot(t,xsol)
plt.figure()
plt.plot(t,usol)
plt.show()
