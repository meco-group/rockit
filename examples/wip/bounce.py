from ocpx import OcpMultiStage, DirectMethod, MultipleShooting
from ocpx.freetime import FreeTime

ocp = OcpMultiStage()

ocp.method(DirectMethod(solver='ipopt'))
stage = ocp.stage(t0=FreeTime(0), T=FreeTime(1))  # Omitting means variable

p = stage.state()
v = stage.state()

stage.set_der(p, v)
stage.set_der(v, -9.81)

stage.subject_to(p <= 5)
stage.method(MultipleShooting(N=20, M=4, intg='cvodes'))

ocp.subject_to(stage.t0 == 0)  # not stage.subject_to !

s = stage

stages = [s]
for i in range(10):
    ocp.subject_to(s.at_t0(p) == 0)
    ocp.subject_to(s.at_tf(p) == 0)

    s_next = ocp.stage(stage)  # copy constructor

    ocp.subject_to(s.tf == s_next.t0)
    # Bouncing inverts (and diminimishes velocity)
    ocp.subject_to(s.at_tf(v) == -0.9 * s_next.at_t0(v))

    s = s_next
    stages.append(s_next)

sol = ocp.solve()

for s in stages:
    ts, ps = sol(s).sample_sim(p)
    # plot(ts, ps)
