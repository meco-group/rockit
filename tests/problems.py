from ocpx import Ocp, DirectMethod, MultipleShooting, FreeTime


def integrator_control_problem(T=1, u_max=1, x0=0, stage_method=None, t0=0):
    if stage_method is None:
      stage_method = MultipleShooting()
    ocp = Ocp(t0=t0, T=T)

    x = ocp.state()
    u = ocp.control()

    ocp.set_der(x, u)

    ocp.subject_to(u <= u_max)
    ocp.subject_to(-u_max <= u)

    ocp.add_objective(ocp.at_tf(x))
    if x0 is not None:
        ocp.subject_to(ocp.at_t0(x) == x0)

    ocp.solver('ipopt')

    ocp.method(stage_method)

    return (ocp, x, u)

def bang_bang_problem(stage_method):
    ocp = Ocp(T=FreeTime(1))

    p = ocp.state()
    v = ocp.state()
    u = ocp.control()

    ocp.set_der(p, v)
    ocp.set_der(v, u)

    ocp.subject_to(u <= 1)
    ocp.subject_to(-1 <= u)

    ocp.add_objective(ocp.T)
    ocp.subject_to(ocp.at_t0(p) == 0)
    ocp.subject_to(ocp.at_t0(v) == 0)
    ocp.subject_to(ocp.at_tf(p) == 1)
    ocp.subject_to(ocp.at_tf(v) == 0)

    ocp.solver('ipopt')

    ocp.method(stage_method)

    return (ocp, ocp.solve(), p, v, u)
    
