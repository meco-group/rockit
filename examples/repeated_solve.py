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
Best practices for repeatedly solved OCPs
==========================================

"""


import numpy as np
from rockit import *
import time

ocp = Ocp(t0=0, T=10)


x1 = ocp.state()
x2 = ocp.state()

u = ocp.control()
e = 1 - x2**2

ocp.set_der(x1, e * x1 - x2 + u)
ocp.set_der(x2, x1)

lagrange = ocp.integral(x1**2 + x2**2 + u**2)

ocp.add_objective(lagrange)
ocp.add_objective(ocp.at_tf(x1**2))

b = ocp.parameter()

ocp.subject_to(x1 >= b)
ocp.subject_to(-1 <= (u <= 1 ))

ocp.subject_to(ocp.at_t0(x1) == 0)
ocp.subject_to(ocp.at_t0(x2) == 1)

silent_options = {}
silent_options["print_time"] = False
silent_options["ipopt.print_level"] = 0
ocp.solver('ipopt', silent_options)

method = MultipleShooting(N=10, intg='rk')
ocp.method(method)

# Solve a bunch of similar ocps in a row,
# the slow way

results_L = []
results_u = []

t0 = time.time()
for b_val in np.linspace(-0.5,-0.25,10):

    # Set a parameter value
    ocp.set_value(b, b_val)

    # Set an initial trajectory for a control
    ocp.set_initial(u, np.linspace(b_val, 1, 10))

    # Set an initial trajectory for a state
    ocp.set_initial(x1, np.linspace(b_val, 1, 11))

    # Solve
    sol = ocp.solve()

    # Retrieve value of non-signal at solution
    L_sol = sol.value(lagrange)
    # Sample signal at the solution
    [_,u_sol] = sol.sample(u,grid='control')

    # Store results
    results_L.append(L_sol)
    results_u.append(u_sol)
print("elapsed time [s]", time.time()-t0)


# Solve a bunch of similar ocps in a row,
# the faster way

prob_solve = ocp.to_function('prob_solve',
    [  # Inputs
        ocp.value(b),
        ocp.sample(u,grid='control-')[1],
        ocp.sample(x1,grid='control')[1]
    ],
    [   # Outputs
        ocp.value(lagrange),
        ocp.sample(u,grid='control')[1]
    ],
    ["b","u_init","x1_init"], # Inputs labels
    ["lag","u"]) # Output labels

# prob_solve is a CasADi Function with 3 inputs and 2 outputs
print(prob_solve)

results2_L = []
results2_u = []

t0 = time.time()
for b_val in np.linspace(-0.5,-0.25,10):
    res = prob_solve(b=b_val,u_init=np.linspace(b_val, 1, 10),x1_init=np.linspace(b_val, 1, 11))

    L_sol = float(res["lag"])
    u_sol = np.array(res["u"]).squeeze()
    # Store results
    results2_L.append(L_sol)
    results2_u.append(u_sol)
print("elapsed time [s]", time.time()-t0)
