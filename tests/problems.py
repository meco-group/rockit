from ocpx import OcpMultiStage, DirectMethod, MultipleShooting


def integrator_control_problem(T, u_max, x0, stage_method, t0=0):
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

    return (ocp.solve(), stage, x, u)
