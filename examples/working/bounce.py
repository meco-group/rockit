"""
Bouncing ball example.

In this example, we want to shoot a ball from the ground up so that after 2
bounces, it will reach the height of 0.5 meter.
"""

from ocpx import OcpMultiStage, DirectMethod, MultipleShooting, FreeTime
import matplotlib.pyplot as plt


def create_bouncing_ball_stage(ocp):
    """
    Create a bouncing ball stage.

    This function creates a stage of a bouncing ball that bounces no higher
    than 5 meters above the ground.

    Returns
    -------
    stage : :obj:`~ocpx.stage.Stage`
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

    stage.subject_to(p <= 5)
    stage.subject_to(p >= 0)
    stage.method(MultipleShooting(N=20, M=1, intg='rk'))

    return stage, p, v


ocp = OcpMultiStage()

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
ts1, ps1 = sol.sample(stage1, p1, grid='control')
ts2, ps2 = sol.sample(stage2, p2, grid='control')
ts3, ps3 = sol.sample(stage3, p3, grid='control')
plt.plot(ts1, ps1)
plt.plot(ts2, ps2)
plt.plot(ts3, ps3)

plt.show(block=True)
