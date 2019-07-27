from ocpx import OcpMultiStage, DirectMethod, MultipleShooting


def integrator_control_problem(T=1, u_max=1, x0=0, stage_method=MultipleShooting(), t0=0):
    ocp = OcpMultiStage()
    stage = ocp.stage(t0=t0, T=T)

    x = stage.state()
    u = stage.control()

    stage.set_der(x, u)

    stage.subject_to(u <= u_max)
    stage.subject_to(-u_max <= u)

    stage.add_objective(stage.at_tf(x))
    stage.subject_to(stage.at_t0(x) == x0)

    ocp.method(DirectMethod(solver='ipopt'))

    stage.method(stage_method)

    return (ocp, ocp.solve(), stage, x, u)

def bang_bang_problem(stage_method):
    ocp = OcpMultiStage()
    stage = ocp.stage(T=ocp.free(1))

    p = stage.state()
    v = stage.state()
    u = stage.control()

    stage.set_der(p, v)
    stage.set_der(v, u)

    stage.subject_to(u <= 1)
    stage.subject_to(-1 <= u)

    stage.add_objective(stage.T)
    stage.subject_to(stage.at_t0(p) == 0)
    stage.subject_to(stage.at_t0(v) == 0)
    stage.subject_to(stage.at_tf(p) == 1)
    stage.subject_to(stage.at_tf(v) == 0)

    ocp.method(DirectMethod(solver='ipopt'))

    stage.method(stage_method)

    return (ocp, ocp.solve(), stage, p, v, u)
    
