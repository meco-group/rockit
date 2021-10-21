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
from casadi import vertcat, sumsqr, interpolant

ocp = Ocp(T=10)

# Bicycle model

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
#ocp.subject_to(ocp.at_tf(x)==0)
#ocp.subject_to(ocp.at_tf(y)==10)

ocp.set_initial(x,0)
ocp.set_initial(y,ocp.t)
ocp.set_initial(theta, pi/2)
ocp.set_initial(V,1)

ocp.subject_to(0 <= (V<=1))
ocp.subject_to( -pi/6 <= (delta<= pi/6))

sdot = ocp.control()

s = ocp.state()

ocp.set_der(s, sdot)
ocp.subject_to(sdot>=0)


tunnel_s = [0,1,2,3,4,5,1000]
tunnel_x = [0,0,-2,-2,0,0,0]
tunnel_y = [0,2,4,6,8,10,1000]
tunnel_r = [2,2,1,1,2,2,2]

N = 20
ocp.set_initial(s, np.linspace(tunnel_s[0],tunnel_s[-2], N))
ocp.set_initial(sdot, (tunnel_s[-2]-tunnel_s[0])/10)

ocp.subject_to(ocp.at_t0(s)==0)

frac = 0.499

spline_x = interpolant('x','bspline',[tunnel_s],tunnel_x,{"algorithm": "smooth_linear","smooth_linear_frac":frac})
spline_y = interpolant('y','bspline',[tunnel_s],tunnel_y,{"algorithm": "smooth_linear","smooth_linear_frac":frac})
spline_r = interpolant('r','bspline',[tunnel_s],tunnel_r,{"algorithm": "smooth_linear","smooth_linear_frac":frac})

ocp.subject_to( (x-spline_x(s))**2+(y-spline_y(s))**2 <= spline_r(s)**2)

# Minimal time
ocp.add_objective(-ocp.at_tf(s))

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N,M=4,intg='rk'))

# solve
sol = ocp.solve()

from pylab import *
figure()

ts, ss = sol.sample(s, grid='integrator',refine=10)
ts = np.linspace(0,2*pi,1000)
xs = np.array(spline_x(ss))
ys = np.array(spline_y(ss))
rs = np.array(spline_r(ss))
for i in range(ss.shape[0]):
  plot(xs[i]+rs[i]*cos(ts),ys[i]+rs[i]*sin(ts),'r-')

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')

plot(xs, ys,'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')


ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)

plot(xs, ys, '-')





axis('equal')
show(block=True)
