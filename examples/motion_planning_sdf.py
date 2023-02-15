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
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#     MA  02110-1301  USA
#
#

"""
Motion planning
===============

Simple motion planning with a signed distance field

"""

from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr, interpolant, DM, horzcat
from pylab import *

# Fixed time problem (go as far as possible within 10s)
ocp = Ocp(T=FreeTime(10.0))
N = 20

# Bicycle model
x = ocp.state()
y = ocp.state()
theta = ocp.state()

delta = ocp.control()
V = ocp.control()

L = 1

p = vertcat(x,y)

ocp.set_der(x, V*cos(theta))
ocp.set_der(y, V*sin(theta))
ocp.set_der(theta, V/L*tan(delta))

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==-1)
ocp.subject_to(ocp.at_t0(y)==6)
ocp.subject_to(ocp.at_t0(theta)==-pi/2)

# Final constraint
ocp.subject_to(ocp.at_tf(x)==12)
ocp.subject_to(ocp.at_tf(y)==6)

# Initial guesses for controls
ocp.set_initial(delta, 0)
ocp.set_initial(V, 1)

# Path constraints
ocp.subject_to(0 <= (V <= 1))
ocp.subject_to(-pi/6 <= (delta <= pi/6))


# Get obstacle bitmap
import imageio
obstacles = np.array(imageio.imread("obstacles.png"))[:,:,1]
obstacles = (obstacles+0.0)/255

# Distance transform
import skfmm # scikit-fmm
sdf = skfmm.distance(obstacles)

# We will assume the image covers the following section of x and y axis
xspan = (-5,15)
yspan = (-5,15)

# Correctly plotting an image on a standard (x,y) coordinate system requires some transformation
sdf = np.flipud(sdf).T

figure()

# Verify image
imshow(sdf.T, origin='lower',extent=(*xspan,*yspan))
colorbar()

# Create a 2D spline form image

d_knots = [list(np.linspace(*xspan,sdf.shape[0])),list(np.linspace(*yspan,sdf.shape[1]))]

d_flat = sdf.ravel(order='F')
    
SDF = interpolant('SDF', 'bspline', d_knots, d_flat,
                       {"algorithm": "smooth_linear",
                        "smooth_linear_frac": 0.4})

# r = np.meshgrid(*d_knots,indexing='ij')
# data = horzcat(r[0].ravel(order='F'),r[1].ravel(order='F')).T
# d = SDF(data).reshape((sdf.shape[0],-1))
# scatter(np.array(data[0,:]),np.array(data[1,:]),1,c=np.array(SDF(data)),marker='.')

ocp.subject_to( SDF(p) >= 100)

# Go as far as possible (i.e. make s as large as possible).
ocp.add_objective(ocp.T)

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N, M=4, intg='rk'))

# solve
sol = ocp.solve()

# Solution
ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')
plot(xs, ys, 'bo')

ts, xs = sol.sample(x, grid='integrator', refine=10)
ts, ys = sol.sample(y, grid='integrator', refine=10)
plot(xs, ys, '-')


axis('equal')

figure()
ts, ps = sol.sample(p,grid='control')

plot(ts,SDF(ps.T).T,'bo')

ts, ps = sol.sample(p,grid='integrator', refine=10)

plot(ts,SDF(ps.T).T,'-')
grid(True)

xlabel("Times [s]")
xlabel("Distance")
title("Minimum distance")

show(block=True)
