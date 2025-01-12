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
Multistage drone race example
=============================

In this example, we use a multistage approach to tackle a gate racing problem
which occurs in drone racing. The first gate is a rectangular gate, the second
is circular. In each stage of the multistage ocp, the collision free space in
front of, through and behind each gate is represented by a linear inequality
constraint.

This approach has been published as:
Bos, M., Decré, W., Swevers, J., Pipeleers, G. (2021).
'Multi-stage Optimal Control Problem Formulation for Drone Racing through Gates and Tunnels',
Proceedings of the 17th IEEE International Conference on Advanced Motion Control (AMC22),
(Paper No. 21). Presented at the 2022 IEEE 17th International Conference on Advanced Motion Control (AMC),
Padova, Italy, 18 Feb 2022-20 Feb 2022
"""


import casadi as c
import rockit as r
import matplotlib.pyplot as plt
from helpers.drone_multistage_helpers import *


# Plot settings
# -------------
draw_quad = True
draw_ocp_ctrl_grid = False

# Hyperparams
# -----------
N1 = 20
N2 = 10
N3 = 20
N4 = 10
N5 = 20
N = (N1, N2, N3, N4, N5)
dT1guess = 1
dT2guess = 0.4
dT3guess = 1
dT4guess = 0.4
dT5guess = 1
T1guess = dT1guess
T2guess = T1guess + dT2guess
T3guess = T2guess + dT3guess
T4guess = T3guess + dT4guess
T5guess = T4guess + dT5guess
Ts = 0.02

# Limits
# ------
atmin = 0
atmax = 9.18*4
datmax = 9.81*5
tiltmax = 1.1
dtiltmax = 6.
dyawmax = 2.

# Initial & terminal conditions
# -----------------------------
p0 = c.vertcat(1., 1., 9.)
v0 = c.vertcat(0, 0, 0)
eul0 = c.vertcat(0, 0, 0)

pf = c.vertcat(9., 9., 5.)
vf = c.vertcat(0, 0, 0)
eulf = c.vertcat(0, 0, 0)

# Room layout
# -----------
room_width = 10
room_depth = 10
room_height = 10

margin = 0.35

# GATE 1 - rectangular
xg1 = 3
yg1 = 3
zg1 = 8
pg1 = c.vertcat(xg1, yg1, zg1)  # Position of gate center
wg1 = 4.  # Gate width
hg1 = 0.7  # Gate height
dg1 = 0.7  # Gate depth
eul_g1 = c.vertcat(0.1, 0.2, 0.2)  # Gate orientation in rpy Euler angles
Rg1, exg1, eyg1, ezg1 = get_rotation_matrix_from_euler(eul_g1)
# GATE 2 - circular
xg2 = 6
yg2 = 5
zg2 = 5
pg2 = c.vertcat(xg2, yg2, zg2)  # Position of gate center
dg2 = 3  # Gate depth
rg2 = 0.6  # Gate radius
eul_g2 = c.vertcat(0., 0.2, 1.5)  # Gate orientation in rpy Euler angles
Rg2, exg2, eyg2, ezg2 = get_rotation_matrix_from_euler(eul_g2)

# Geometry of boxes
# Approach room 1
w1 = get_w_approach(pg1, exg1, dg1, margin)
# Through room 1
w2 = get_w_through(pg1, eyg1, ezg1, hg1, wg1, margin)
# Leave room 1 = Approach room 2
w3 = get_w_flyaway_approach(pg1, exg1, dg1, pg2, exg2, dg2, margin)
# Through room 2 (circular gate)
n4 = exg2
# Fly away room 2
w5 = get_w_flyaway(pg2, exg2, dg2, margin)


# -----------------------------------------------
# Ocp construction functions
def create_stage(ocp, t0, T, N, w):
    '''Create a rockit.stage object with drone dynamics, hyperplane
    constraints for this specific stage, method (multiple shooting) and
    contribution to the objective (total stage time).
    '''
    stage = ocp.stage(t0=t0, T=T)

    # States
    p = stage.state(3)
    v = stage.state(3)
    # Control inputs
    at = stage.control(order=1)
    eul = stage.control(3, order=1)
    (phi, theta, psi) = (eul[0], eul[1], eul[2])
    Rrpy, _, _, _ = get_rotation_matrix_from_euler(eul)

    # Dynamics
    at_ = Rrpy @ c.vertcat(0, 0, at)
    g = c.vertcat(0, 0, -9.81)
    a = at_ + g

    stage.set_der(p, v)
    stage.set_der(v, a)

    # State and control input constraints
    stage.subject_to(-np.pi/2 <= (phi <= np.pi/2))
    stage.subject_to(-np.pi/2 <= (theta <= np.pi/2))

    stage.subject_to(c.cos(theta)*c.cos(phi) >= c.cos(tiltmax))
    stage.subject_to(-dtiltmax <= (stage.der(phi) <= dtiltmax))
    stage.subject_to(-dtiltmax <= (stage.der(theta) <= dtiltmax))
    stage.subject_to(psi == 0)
    stage.subject_to(-dyawmax <= (stage.der(psi) <= dyawmax))

    stage.subject_to(atmin <= (at <= atmax))
    stage.subject_to(-datmax <= (stage.der(at) <= datmax))

    # Hyperplane constraint
    phom = c.vertcat(p, 1)
    stage.subject_to(w.T @ phom <= 0)

    # Method
    stage.method(r.MultipleShooting(N=N, M=1, intg='rk'))
    # Objective
    stage.add_objective(stage.T)

    return stage, p, v, at_, at, eul, Rrpy


def create_tube_stage(ocp, t0, T, N, pg, ng, rg, m):
    '''Same as create_stage, but with tubular constraint instead of hyperplane
    constraint.
    '''
    stage = ocp.stage(t0=t0, T=T)

    # States
    p = stage.state(3)
    v = stage.state(3)
    # Control inputs
    at = stage.control(order=1)
    eul = stage.control(3, order=1)
    (phi, theta, psi) = (eul[0], eul[1], eul[2])
    Rrpy, _, _, _ = get_rotation_matrix_from_euler(eul)

    # Dynamics
    at_ = Rrpy @ c.vertcat(0, 0, at)
    g = c.vertcat(0, 0, -9.81)
    a = at_ + g

    stage.set_der(p, v)
    stage.set_der(v, a)

    # State and control input constraints
    stage.subject_to(-np.pi/2 <= (phi <= np.pi/2))
    stage.subject_to(-np.pi/2 <= (theta <= np.pi/2))

    stage.subject_to(c.cos(theta)*c.cos(phi) >= c.cos(tiltmax))
    stage.subject_to(-dtiltmax <= (stage.der(phi) <= dtiltmax))
    stage.subject_to(-dtiltmax <= (stage.der(theta) <= dtiltmax))
    stage.subject_to(psi == 0)
    stage.subject_to(-dyawmax <= (stage.der(psi) <= dyawmax))

    stage.subject_to(atmin <= (at <= atmax))
    stage.subject_to(-datmax <= (stage.der(at) <= datmax))

    # Tubular constraint
    dp = pg - p
    dp_par = (dp.T @ ng) * ng
    dp_perp = dp - dp_par
    stage.subject_to(dp_perp.T @ dp_perp <= (rg - m)**2)

    # Method
    stage.method(r.MultipleShooting(N=N, M=1, intg='rk'))
    # Objective
    stage.add_objective(stage.T)

    return stage, p, v, at_, at, eul, Rrpy


def stitch_stages(ocp, stage1, stage2):
    '''Set equality constraints to the end time and state of stage1 and the
    initial time and state of stage2 to stitch them together.
    '''
    # Stitch time
    ocp.subject_to(stage1.tf == stage2.t0)
    # Stitch states
    for i in range(len(stage1.states)):
        ocp.subject_to(stage2.at_t0(stage2.states[i])
                       == stage1.at_tf(stage1.states[i]))


# =============================================================================
# Main - setting up the OCP
ocp = r.Ocp()

# Stage 1 - Approach
stage1, p1, v1, at1_, at1, eul1, R1 = create_stage(
                                       ocp, r.FreeTime(0), r.FreeTime(T1guess),
                                       N1, w1)
ocp.subject_to(stage1.t0 == 0)
ocp.subject_to(stage1.at_t0(p1) == p0)
ocp.subject_to(stage1.at_t0(v1) == v0)
ocp.subject_to(stage1.at_t0(eul1) == eul0)

# Stage 2 - Through
stage2, p2, v2, at2_, at2, eul2, R2 = create_stage(
                                 ocp, r.FreeTime(T1guess), r.FreeTime(T2guess),
                                 N2, w2)
stitch_stages(ocp, stage1, stage2)

# Stage 3 - Leave / approach
stage3, p3, v3, at3_, at3, eul3, R3 = create_stage(
                                 ocp, r.FreeTime(T2guess), r.FreeTime(T3guess),
                                 N3, w3)
stitch_stages(ocp, stage2, stage3)

# Stage 4 - Through 2
stage4, p4, v4, at4_, at4, eul4, R4 = create_tube_stage(
                                 ocp, r.FreeTime(T3guess), r.FreeTime(T4guess),
                                 N4, pg2, n4, rg2, margin)
stitch_stages(ocp, stage3, stage4)

# Stage 5 - Leave 2
stage5, p5, v5, at5_, at5, eul5, R5 = create_stage(
                                 ocp, r.FreeTime(T4guess), r.FreeTime(T5guess),
                                 N5, w5)
stitch_stages(ocp, stage4, stage5)

# Terminal conditions
ocp.subject_to(stage5.at_tf(p5) == pf)
ocp.subject_to(stage5.at_tf(v5) == vf)
ocp.subject_to(stage5.at_tf(eul5) == eulf)

# Solver options
opts = {"expand": True,
        "verbose": False,
        "print_time": True,
        "error_on_fail": True,
        "ipopt": {"linear_solver": "mumps",  # "ma57" is faster!
                  "max_iter": 1000,
                  'print_level': 5,
                  'sb': 'yes',  # Suppress IPOPT banner
                  'tol': 1e-4,
                  'hessian_approximation': 'limited-memory'
                  }}
ocp.solver("ipopt", opts)


# Create sampler
sampler1 = stage1.sampler([p1, v1, at1, eul1, R1])
sampler2 = stage2.sampler([p2, v2, at2, eul2, R2])
sampler3 = stage3.sampler([p3, v3, at3, eul3, R3])
sampler4 = stage4.sampler([p4, v4, at4, eul4, R4])
sampler5 = stage5.sampler([p5, v5, at5, eul5, R5])
samplers = (sampler1, sampler2, sampler3, sampler4, sampler5)

dT1 = ocp.value(stage1.T)
dT2 = ocp.value(stage2.T)
dT3 = ocp.value(stage3.T)
dT4 = ocp.value(stage4.T)
dT5 = ocp.value(stage5.T)

T1 = dT1
T2 = T1 + dT2
T3 = T2 + dT3
T4 = T3 + dT4
T5 = T4 + dT5

# Create a CasADi function
solve_ocp = ocp.to_function(
    'solve_ocp',
    [ocp._method.opti.x],
    [T1, T2, T3, T4, T5, ocp._method.opti.x, ocp.gist])

# =============================================================================
# Solve the OCP!

prev_sol = 0
T1, T2, T3, T4, T5, prev_sol, gist = solve_ocp(prev_sol)
# Solve again using previous solution:
# T1, T2, T3, T4, T5, prev_sol, gist = solve_ocp(prev_sol)

# =============================================================================
# Postprocess
T = (float(T1), float(T2), float(T3), float(T4), float(T5))
(T1, T2, T3, T4, T5) = T

# Control rate (Ts) sampling
(psol_ctrl, vsol_ctrl, atsol_ctrl, eulsol_ctrl,
    Rsol_ctrl, t_ctrl) = get_sol_ctrl_grid(gist, samplers, Ts, T)

# OCP control grid sampling
(psol, vsol, atsol, eulsol,
    Rsol, t, sol) = get_sol_OCP_grid(gist, samplers, T, N)

# =============================================================================
# Plots
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X-Position (m)')
ax.set_ylabel('Y-Position (m)')
ax.set_zlabel('Z-Position (m)')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Plot gates
alpha = 0.5
plot_gate(pg1, Rg1, wg1, hg1, dg1, ax, alpha)
plot_circular_gate(pg2.full(), Rg2.full(), rg2, dg2, ax, alpha)

# Plot quad
if draw_quad:
    step = 7
    plot_quad3d(psol_ctrl[:, 0], psol_ctrl[:, 1], psol_ctrl[:, 2], Rsol_ctrl,
                ax, [0, room_width, 0, room_depth, 0, room_height], step)

# Plot path
if draw_ocp_ctrl_grid:
    (p1sol, p2sol, p3sol, p4sol, p5sol) = (sol[0][0], sol[1][0], sol[2][0],
                                           sol[3][0], sol[4][0])
    ax.plot(p1sol[:, 0], p1sol[:, 1], p1sol[:, 2],
            '-o', color=red)
    ax.plot(p2sol[:, 0], p2sol[:, 1], p2sol[:, 2],
            '-o', color=yellow)
    ax.plot(p3sol[:, 0], p3sol[:, 1], p3sol[:, 2],
            '-o', color=blue)
    ax.plot(p4sol[:, 0], p4sol[:, 1], p4sol[:, 2],
            '-o', color=yellow)
    ax.plot(p5sol[:, 0], p5sol[:, 1], p5sol[:, 2],
            '-o', color=blue)

plt.show(block=True)
