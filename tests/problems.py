from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime


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

    return (ocp, p, v, u)


def vdp_dae(method,x1limit=True):
  ocp = Ocp(T=10)

  # Define 2 states
  x1 = ocp.state()
  x2 = ocp.state()

  z = ocp.algebraic()

  # Define 1 control
  u = ocp.control(order=0)

  # Specify ODE
  ocp.set_der(x1, z * x1 - x2 + u)
  ocp.set_der(x2, x1)
  ocp.add_alg(z-(1 - x2**2))

  # Lagrange objective
  ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))

  # Path constraints
  ocp.subject_to(-1 <= (u <= 1))
  if x1limit:
    ocp.subject_to(x1 >= -0.25)

  ocp.subject_to(ocp.at_tf(x1) >= 0)

  # Initial constraints
  ocp.subject_to(ocp.at_t0(x1) == 0)
  ocp.subject_to(ocp.at_t0(x2) == 1)

  # Pick an NLP solver backend
  ocp.solver('ipopt')

  # Pick a solution method
  ocp.method(method)
  return (ocp, x1, x2, u)

def vdp(method,grid='control'):
  ocp = Ocp(T=10)

  # Define 2 states
  x1 = ocp.state()
  x2 = ocp.state()

  # Define 1 control
  u = ocp.control(order=0)

  # Specify ODE
  ocp.set_der(x1, (1 - x2**2) * x1 - x2 + u)
  ocp.set_der(x2, x1)

  # Lagrange objective
  ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))

  # Path constraints
  ocp.subject_to(-1 <= (u <= 1))
  ocp.subject_to(x1 >= -0.25, grid=grid)

  # Initial constraints
  ocp.subject_to(ocp.at_t0(x1) == 0)
  ocp.subject_to(ocp.at_t0(x2) == 1)

  # Pick an NLP solver backend
  ocp.solver('ipopt')

  # Pick a solution method
  ocp.method(method)
  return (ocp, x1, x2, u)
