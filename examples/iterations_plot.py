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
Iterations plotting
===================

Example of making plots while the solver converges
"""

from rockit import *

ocp = Ocp(T=10)

# Define 2 states
x1 = ocp.state()
x2 = ocp.state()

# Define 1 control
u = ocp.control(order=0)

# Specify ODE
ocp.set_der(x1, (1 - x2**2) * x1 - x2 + u)
ocp.set_der(x2, x1)

# Lagrange objective
ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))

# Path constraints
ocp.subject_to(-1 <= (u <= 1))
ocp.subject_to(x1 >= -0.25) # grid='inf')

# Initial constraints
ocp.subject_to(ocp.at_t0(x1) == 0)
ocp.subject_to(ocp.at_t0(x2) == 1)

# Pick an NLP solver backend
ocp.solver('ipopt')

# Pick a solution method
method = MultipleShooting(N=10, M=1, intg='rk')
ocp.method(method)

from pylab import *
def plotme(iter, sol):
  figure(figsize=(15, 4))
  for state, label, color in [(x1,'x1','r'),(x2,'x2','g'),(u,'u','b')]:
    tsa, x1a = sol.sample(state, grid='control')
    tsb, x1b = sol.sample(state, grid='integrator')
    tsc, x1c = sol.sample(state, grid='integrator', refine=100)
    plot(tsc, x1c, color+'-',label=label)
    plot(tsa, x1a, color+'o')
    plot(tsb, x1b, color+'.')
  xlabel("Times [s]")
  ylim([-0.3,1.1])
  title('Iteration %d' % iter)
  legend()
  grid(True)

ocp.callback(plotme)

# Solve
sol = ocp.solve()

# sphinx_gallery_thumbnail_number = 2

show(block=True)