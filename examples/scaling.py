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
A scaling Example
===================

Shows how to fix scaling of a badly sclaed problem
"""


# Make available sin, cos, etc
from numpy import *
# Import the project
from rockit import *
#%%

# In this file, we start from the Vanderpol oscillator example of hello world
# The following is the expected solution trajectory:
x1_ref = [ 2.52469687e-36, -2.50000010e-01, -2.50000009e-01 ,-2.50000006e-01,
 -1.48309122e-01, -3.72646090e-02,  6.56395958e-03,  1.06163067e-02,
  5.16887935e-03,  1.71826868e-03, -2.80386510e-04]
x2_ref = [ 1.00000000e+00, 8.68298025e-01 , 6.04317331e-01 , 3.36762795e-01,
  1.11997054e-01 , 3.60719567e-04 ,-2.04611315e-02, -1.14118570e-02,
 -2.26223705e-03 , 1.82764512e-03 , 2.79793225e-03]
u_ref = [7.19549789e-01,  8.55058241e-01 , 6.76514569e-01,  5.29132015e-01,
  2.68151632e-01 , 5.06393975e-02, -2.12432317e-02, -2.09098487e-02,
 -7.43432715e-03, -4.75629541e-04 ,-4.75629541e-04]


# Different from hello world, we choose to model the system using inappropriate units (km/s, mm, m) that give rise to both very small and large magnitudes of decision variables and constraints

# If we don't take further action (with_scaling = False), the resultant OCP will fail to solve.

with_scaling = True

ocp = Ocp(t0=0, T=10)

x1 = ocp.state(scale=1/1000 if with_scaling else 1) # [km/s] - nominal value is 1m/s = 1/1000 km/s
# In essence, the scale argument has the same effect as defining:
# x1 = 1/1000*ocp.state(),
# Except that ocp.set_initial(x1, ...) keeps working

x2 = ocp.state(scale=1000 if with_scaling else 1)   # [mm]   - nominal value is 1m = 1000 mm

u = ocp.control(scale=3600**2 if with_scaling else 1) # [m/h^2] - nominal value is 1 m/s^2 = 3600^2 m/h^2

# Physics expressed in SI units and converted back to dimensional derivative
ocp.set_der(x1, ((1 - 1e-6*x2**2) * 1000*x1 - 1e-3*x2 + u/3600**2)/1000, scale=1/1000 if with_scaling else 1)
ocp.set_der(x2, 1e6*x1, scale=1000 if with_scaling else 1)

ocp.add_objective(ocp.integral((1000*x1)**2 + (1e-3*x2)**2 + (u/3600**2)**2))
ocp.add_objective(ocp.at_tf((1000*x1)**2))

# Path constraints
ocp.subject_to(x1 >= -0.25*1e-3, scale=1/1000 if with_scaling else 1)
# The scale argument has the same effect as multiplying all sides with 1000
ocp.subject_to(-3600**2 <= (u <= 3600**2), scale=3600**2 if with_scaling else 1)

# Boundary constraints
ocp.subject_to(ocp.at_t0(x1) == 0, scale=1/1000 if with_scaling else 1)
ocp.subject_to(ocp.at_t0(x2) == 1000, scale=1000 if with_scaling else 1)


# Once the problem is properly scale, it is best to shut off Ipopt scaling
options = {"ipopt.nlp_scaling_method": "none"}
# It can also make sense to shut off auto-scaling of Mumps linear solver
# In fact, this sometimes makes Mumps more resilient like ma57
options["ipopt.mumps_permuting_scaling"] = 0
options["ipopt.mumps_scaling"] = 0

ocp.solver('ipopt', options)

method = MultipleShooting(N=10, intg='rk')
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
tsa, x1a = sol.sample(1000*x1, grid='control')
tsa, x2a = sol.sample(1e-3*x2, grid='control')

tsb, x1b = sol.sample(1000*x1, grid='integrator')
tsb, x2b = sol.sample(1e-3*x2, grid='integrator')


print("x1",x1a)
print("x2",x2a)

assert np.linalg.norm(x1a-np.array(x1_ref))<1e-7
assert np.linalg.norm(x2a-np.array(x2_ref))<1e-7


figure(figsize=(10, 4))
subplot(1, 2, 1)
plot(tsb, x1b, '.-')
plot(tsa, x1a, 'o')
xlabel("Times [s]", fontsize=14)
grid(True)
title('State x1')

subplot(1, 2, 2)
plot(tsb, x2b, '.-')
plot(tsa, x2a, 'o')
legend(['grid_integrator', 'grid_control'])
xlabel("Times [s]", fontsize=14)
title('State x2')
grid(True)

# sphinx_gallery_thumbnail_number = 2

tsol, usol = sol.sample(u/3600**2, grid='control')
print("u",usol)
assert np.linalg.norm(usol-np.array(u_ref))<1e-7


# Refine the grid for a more detailed plot
tsol, usol = sol.sample(u/3600**2, grid='integrator', refine=100)

figure()
plot(tsol,usol)
title("Control signal")
xlabel("Times [s]")
grid(True)

tsc, x1c = sol.sample(1000*x1, grid='integrator', refine=100)

figure(figsize=(15, 4))
plot(tsc, x1c, '-')
plot(tsa, x1a, 'o')
plot(tsb, x1b, '.')
xlabel("Times [s]")
grid(True)

show(block=True)
