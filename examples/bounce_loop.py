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
Bouncing ball example (using for loop)
======================================

In this example, we want to shoot a ball from the ground up so that after 2
bounces, it will reach the height of 0.5 meter.
"""

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime
import matplotlib.pyplot as plt

ocp = Ocp()

stage = ocp.stage(t0=FreeTime(0), T=FreeTime(1))
p = stage.state()
v = stage.state()

stage.set_der(p, v)
stage.set_der(v, -9.81)

stage.subject_to(stage.at_t0(v) >= 0)
stage.subject_to(p >= 0)
stage.method(MultipleShooting(N=1, M=20, intg='rk'))

stage.subject_to(stage.at_t0(p) == 0)

ocp.subject_to(stage.t0 == 0)

stage_prev = stage

n_bounce = 2
for i in range(n_bounce):
    ocp.subject_to(stage.at_tf(p) == 0)
    stage = ocp.stage(stage_prev)
    ocp.subject_to(stage.at_t0(v) == -0.9 * stage_prev.at_tf(v))
    ocp.subject_to(stage.t0 == stage_prev.tf)

    stage_prev = stage

# Final bounce should reach 0.5 exactly
ocp.subject_to(stage.at_tf(v) == 0)
ocp.subject_to(stage.at_tf(p) == 0.5)

ocp.solver('ipopt')

# Solve
sol = ocp.solve()

# Plot the 3 bounces
plt.figure()

for s in ocp.iter_stages():
    ts, ps = sol(s).sample(p, grid='integrator')
    plt.plot(ts, ps)

plt.show(block=True)
