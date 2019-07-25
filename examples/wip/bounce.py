from ocpx import OcpMultiStage, DirectMethod, MultipleShooting
import matplotlib.pyplot as plt

ocp = OcpMultiStage()

ocp.method(DirectMethod(solver='ipopt'))
stage = ocp.stage(t0=ocp.free(0), T=ocp.free(1))  # Omitting means variable

p = stage.state()
v = stage.state()

stage.set_der(p, v)
stage.set_der(v, -9.81)

stage.subject_to(p <= 5)
stage.method(MultipleShooting(N=20, M=4, intg='cvodes'))

ocp.subject_to(stage.t0 == 0)  # not stage.subject_to !

stages = [stage]
for i in range(3):
    ocp.subject_to(stage.at_t0(p) == 0)
    ocp.subject_to(stage.at_tf(p) == 0)

    stage_next = ocp.stage(stage)  # copy constructor

    ocp.subject_to(stage.tf == stage_next.t0)
    # Bouncing inverts (and diminimishes velocity)
    ocp.subject_to(stage.at_tf(v) == -0.9 * stage_next.at_t0(v))

    stage = stage_next
    stages.append(stage_next)

sol = ocp.solve()
for s in stages[0:-1]:
    ts, ps = sol.sample(s, p, grid=stage.grid_control)
    plt.plot(ts, ps)

plt.show(block=True)
