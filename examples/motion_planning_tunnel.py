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

Simple motion planning within corridor (tunnel)
"""

from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr, interpolant, DM
from pylab import *

# Fixed time problem (go as far as possible within 10s)
ocp = Ocp(T=10)
N = 20

# Bicycle model
x = ocp.state()
y = ocp.state()
theta = ocp.state()

delta = ocp.control()
V = ocp.control()

L = 1

ocp.set_der(x, V*cos(theta))
ocp.set_der(y, V*sin(theta))
ocp.set_der(theta, V/L*tan(delta))

# Initial constraints
ocp.subject_to(ocp.at_t0(x) == 0)
ocp.subject_to(ocp.at_t0(y) == 0)
ocp.subject_to(ocp.at_t0(theta) == pi/2)

# Initial guesses for states
x_init = vertcat(np.linspace(0, -2, int(N/2)+1), np.linspace(-2, 0, int(N/2)))
y_init = np.linspace(0, 10, N+1)
theta_init = np.linspace(pi/2, pi/2, N+1)
ocp.set_initial(x, x_init)  # centerline of tunnel
ocp.set_initial(y, y_init)  # centerline of tunnel
ocp.set_initial(theta, theta_init)

# Initial guesses for controls
ocp.set_initial(delta, 0)
ocp.set_initial(V, 1)

# Path constraints
ocp.subject_to(0 <= (V <= 1))
ocp.subject_to(-pi/6 <= (delta <= pi/6))

# Define a progress variable s as a state. The derivative of this state, sdot,
# is defined as a control input. sdot must be larger than zero in order to
# only have an increasing s (only positive progress along s). Within two
# consecutive values, s might not change too much to avoid going out of the
# predefined circles (upper constraint on sdot).
s = ocp.state()
sdot = ocp.control()
ocp.set_der(s, sdot)
ocp.subject_to(sdot >= 0)
ocp.subject_to(sdot <= 0.5)

# Define a grid for s on which you will define values for the tunnel (and
# extend it to a high value since we do not know yet which value s will take
# at_tf, might be above 5).
s_grid = [0, 1, 2, 3, 4, 5, 1000]

# Define values for x, y and radius of the tunnel along the grid s.
x_grid = [0, 0, -2, -2, 0, 0, 0]
y_grid = [0, 2, 4, 6, 8, 10, 1000]
r_grid = [2, 2, 1, 1, 2, 2, 2]

# Initial guesses for progress variable and its derivative.
ocp.set_initial(s, np.linspace(s_grid[0], s_grid[-2], N+1))
ocp.set_initial(sdot, (s_grid[-2] - s_grid[0])/10)

# Progress state variable s should start at 0.
ocp.subject_to(ocp.at_t0(s) == 0)

# Interpolant parameters;
#     name      label for the resulting Function
#     solver    name of the plugin (linear or bspline)
#     grid    	collection of 1D grids whose outer product defines the full
#                 N-D rectangular grid
#     values	flattened vector of all values for all gridpoints
#     options   linear:
#                   - lookup_mode: Sets, for each grid dimenion, the lookup
#                     algorithm used to find the correct index. 'linear' uses
#                     a for-loop + break; 'exact' uses floored division (only
#                     for uniform grids).
#               bspline:
#                   - algorithm: Algorithm used for fitting the data:
#                     'not_a_knot' (default, same as Matlab), 'smooth_linear'.
#                   - degree: Sets, for each grid dimension, the degree of the
#                     spline.
#                   - linear_solver: Solver used for constructing the
#                     coefficient tensor.
#                   - smooth_linear_frac: When 'smooth_linear' algorithm is
#                     active, determines sharpness between 0 (sharp, as linear
#                     interpolation) and 0.5 (smooth). Default value is 0.1.
frac = 0.499
spline_x = interpolant('x', 'bspline', [s_grid], x_grid,
                       {"algorithm": "smooth_linear",
                        "smooth_linear_frac": frac})
spline_y = interpolant('y', 'bspline', [s_grid], y_grid,
                       {"algorithm": "smooth_linear",
                        "smooth_linear_frac": frac})
spline_r = interpolant('r', 'bspline', [s_grid], r_grid,
                       {"algorithm": "smooth_linear",
                        "smooth_linear_frac": frac})

# Constrain the bicycle position (x, y) to be within a certain distance from
# the centerline of a tunnel, defined by circles with radius spline_r(s)
# around each point on the centerline given by spline_x(s) and spline_y(s).
ocp.subject_to((x-spline_x(s))**2+(y-spline_y(s))**2 <= spline_r(s)**2)

# Go as far as possible (i.e. make s as large as possible).
ocp.add_objective(-ocp.at_tf(s))

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N, M=4, intg='rk'))

# solve
sol = ocp.solve()

# Solution
figure()
ts, ss = sol.sample(s, grid='control')
ts = np.linspace(0, 2*pi, 1000)
xs = np.array(spline_x(ss))
ys = np.array(spline_y(ss))
rs = np.array(spline_r(ss))
for i in range(ss.shape[0]):
    plot(xs[i]+rs[i]*cos(ts), ys[i]+rs[i]*sin(ts), 'r-')
    plot(xs[i], ys[i], 'k.')

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')
plot(xs, ys, 'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')
plot(xs, ys, 'b.')

ts, xs = sol.sample(x, grid='integrator', refine=10)
ts, ys = sol.sample(y, grid='integrator', refine=10)
plot(xs, ys, '-')

axis('equal')
show(block=True)
