from rockit import Ocp, DirectMethod, MultipleShooting

ocp = Ocp()

# Rocket example

stage = ocp.stage(t0=0, T=1)  # Omitting means variable


p = stage.state()  # Position
v = stage.state()  # Velocity
m = stage.control()  # Mass

u = stage.control()  # Thrust

stage.set_der(p, v)
stage.set_der(v, (u - 0.05 * v * v) / m)
stage.set_der(m, -0.1 * u * u)

# Regularize the control
stage.add_objective(stage.integral(u**2))

# Path constraints
stage.subject_to(u >= 0)
stage.subject_to(u <= 0.5)

# Initial constraints
stage.subject_to(stage.at_t0(p) == 0)
stage.subject_to(stage.at_t0(v) == 0)
stage.subject_to(stage.at_t0(m) == 1)

# Final constraints
stage.subject_to(stage.at_tf(p) == 10)
stage.subject_to(stage.at_tf(v) == 0)

ocp.method(DirectMethod(solver='ipopt'))

stage.method(MultipleShooting(N=50, M=2, intg='rk'))

# Missing: a nonzero initial guess for m

sol = ocp.solve()
