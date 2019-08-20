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
Bouncing ball example (simple)
==============================

In this example, we want to shoot a ball from the ground up so that after 2
bounces, it will reach the height of 0.5 meter.
"""

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime
import matplotlib.pyplot as plt


def create_bouncing_ball_stage(ocp):
    """
    Create a bouncing ball stage.

    This function creates a stage of a bouncing ball that bounces no higher
    than 5 meters above the ground.

    Returns
    -------
    stage : :obj:`~rockit.stage.Stage`
        An ocp stage describing the bouncing ball
    p : :obj:`~casadi.MX`
        position variable
    v : :obj:`~casadi.MX`
        velocity variable
    """
    stage = ocp.stage(t0=FreeTime(0), T=FreeTime(1))

    p = stage.state()
    v = stage.state()

    stage.set_der(p, v)
    stage.set_der(v, -9.81)

    stage.subject_to(stage.at_t0(v) >= 0)
    stage.subject_to(p >= 0)
    stage.method(MultipleShooting(N=1, M=20, intg='rk'))

    return stage, p, v


ocp = Ocp()

# Shoot up the ball
stage1, p1, v1 = create_bouncing_ball_stage(ocp)
ocp.subject_to(stage1.t0 == 0)  # Stage starts at time 0
ocp.subject_to(stage1.at_t0(p1) == 0)
ocp.subject_to(stage1.at_tf(p1) == 0)

# After bounce 1
stage2, p2, v2 = create_bouncing_ball_stage(ocp)
ocp.subject_to(stage2.at_t0(v2) == -0.9 * stage1.at_tf(v1))
ocp.subject_to(stage2.at_t0(p2) == stage1.at_tf(p1))
ocp.subject_to(stage2.t0 == stage1.tf)
ocp.subject_to(stage2.at_tf(p2) == 0)

# After bounce 2
stage3, p3, v3 = create_bouncing_ball_stage(ocp)
ocp.subject_to(stage3.at_t0(v3) == -0.9 * stage2.at_tf(v2))
ocp.subject_to(stage3.at_t0(p3) == stage2.at_tf(p2))
ocp.subject_to(stage3.t0 == stage2.tf)
ocp.subject_to(stage3.at_tf(v3) == 0)
ocp.subject_to(stage3.at_tf(p3) == 0.5)  # Stop at a half meter!

ocp.solver('ipopt')

# Solve
sol = ocp.solve()

# Plot the 3 bounces
plt.figure()
ts1, ps1 = sol(stage1).sample(p1, grid='integrator')
ts2, ps2 = sol(stage2).sample(p2, grid='integrator')
ts3, ps3 = sol(stage3).sample(p3, grid='integrator')
plt.plot(ts1, ps1)
plt.plot(ts2, ps2)
plt.plot(ts3, ps3)

plt.show(block=True)
