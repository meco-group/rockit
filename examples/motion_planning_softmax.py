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

from pylab import *
from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, horzcat, sumsqr, vec,sum1,mmax,DM
import casadi as ca

ocp = Ocp(T=FreeTime(10.0))

softmax = False
outer = False
squared = False

# Bicycle model

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

delta = ocp.control()
V     = ocp.control(order=1)

N = 100
L = 0.1

ocp.set_der(x, V*cos(theta))
ocp.set_der(y, V*sin(theta))
ocp.set_der(theta, V/L*tan(delta))

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(ocp.at_t0(y)==0)
ocp.subject_to(ocp.at_t0(theta)==pi/2)
ocp.subject_to(ocp.at_t0(V)==0)

# Final constraint
ocp.subject_to(ocp.at_tf(x)==10)
ocp.subject_to(ocp.at_tf(y)==10)

if not outer:
  ocp.set_initial(x,ocp.t)
ocp.set_initial(y,ocp.t)
ocp.set_initial(theta, pi/2)
ocp.set_initial(V,1)

ocp.subject_to(0 <= (V<=1))
ocp.subject_to(-0.3 <=(ocp.der(V)<=0.3))
ocp.subject_to( -pi/6 <= (delta<= pi/6))

Nobs = 5

# Round obstacle

grid = linspace(3,7,Nobs)
X,Y = meshgrid(grid,grid)
X+=1e-5
r0   = (grid[1]-grid[0])/2*0.75


# 1/alpha*log(sum1(exp(alpha*e)))
#    e has a dominant term:  1/alpha*log(exp(alpha*m)) = m
#    all e equal: 1/alpha*log(N*exp(m*alpha)) = 1/alpha*(log(N)+m*alpha)=m+log(N)/alpha

max_approx = 0.05 # [m]
Nactive = Nobs**2 # Guess how many constraints will be active simultaneously
Nactive = 2

try:
    # Casadi >=3.6
    ca.logsumexp
    def smooth_max(e): return ca.logsumexp(e,max_approx)
    def smooth_min(e): return -ca.logsumexp(e,max_approx)
    
except:
    def basic(e):
      m = mmax(e)
      return m+log(sum1(exp(e-m)))

    alpha = log(Nactive)/max_approx
    def smooth_max(e):
      return 1/alpha*basic(alpha*e)
    def smooth_min(e):
      return -smooth_max(-e)

p = vertcat(x,y)

if softmax:
  dist = sqrt((X-x)**2+(Y-y)**2)-r0
  ocp.subject_to(smooth_min(vec(dist))>=0)
else:
  for i in range(Nobs):
    for j in range(Nobs):
      p0=vertcat(X[i,j],Y[i,j])

      if squared:
        ocp.subject_to(sumsqr(p-p0)>=r0**2)
      else:
        ocp.subject_to(sqrt(sumsqr(p-p0))>=r0)

# Minimal time
ocp.add_objective(ocp.T)

# Pick a solution method
ocp.solver('ipopt',{"expand":True})

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N,M=1,intg='rk'))

# solve
sol = ocp.solve()

#ocp.spy()

from pylab import *
figure()

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')

plot(xs, ys,'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')


ts, xs = sol.sample(x, grid='integrator',refine=100)
ts, ys = sol.sample(y, grid='integrator',refine=100)

plot(xs, ys, '-')

ts = np.linspace(0,2*pi,1000)
for i in range(Nobs):
  for j in range(Nobs):
    plot(X[i,j]+r0*cos(ts),Y[i,j]+r0*sin(ts),'r-')

axis('equal')

show(block=True)
