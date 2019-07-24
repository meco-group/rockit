from ocpx import *

# Inspired from https://github.com/casadi/casadi/blob/master/docs/examples/python/direct_multiple_shooting.py

ocp = OcpMultiStage()

stage = ocp.stage(t0=0,T=10)

# Define states
x1 = stage.state()
x2 = stage.state()

# Defince controls
u = stage.control(1,order=2)

# Specify ODE
stage.set_der(x1, (1-x2**2)*x1 - x2 + u)
stage.set_der(x2,  x1)

# Lagrange objective
stage.add_objective(stage.integral(x1**2 + x2**2 + u**2))

# Path constraints
stage.subject_to(u<=1)
stage.subject_to(-1<=u)
stage.subject_to(x1>=-0.25)

# Initial constraints
stage.subject_to(stage.at_t0(x1)==0)
stage.subject_to(stage.at_t0(x2)==1)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=20,M=4,intg='rk'))

# solve
sol = ocp.solve()
