from ocpx import *
from casadi import *
from casadi.tools import *

import matplotlib.pyplot as plt

ocp = OcpMultiStage()

stage = ocp.stage(t0=0,T=1)

# Problem parameters
M = 1.0
m = 0.1
l = 0.8
g = 9.81
nx = 4
nu = 1

# Define states
X = stage.state(nx)

# Defince controls
F = stage.control(nu,order=1)

# Define parameter
X_0 = stage.parameter(nx);


# X[0] <-- pos
# X[1] <-- dpos
# X[2] <-- theta
# X[3] <-- dtheta

# Specify ODE
# stage.set_der(X[0], X[1])
# stage.set_der(X[1], (-m*l*sin(X[2])*X[3]*X[3] + m*g*cos(X[2])*sin(X[2])+F)/(M + m - m*cos(X[2])*cos(X[2])))
# stage.set_der(X[2], X[3])
# stage.set_der(X[3], (-m*l*cos(X[2])*sin(X[2])*X[3]*X[3] + F*cos(X[2])+(M+m)*g*sin(X[2]))/(l*(M + m - m*cos(X[2])*cos(X[2]))))

stage.set_der(X, vertcat( X[1], (-m*l*sin(X[2])*X[3]*X[3] + m*g*cos(X[2])*sin(X[2])+F)/(M + m - m*cos(X[2])*cos(X[2])), X[3], (-m*l*cos(X[2])*sin(X[2])*X[3]*X[3] + F*cos(X[2])+(M+m)*g*sin(X[2]))/(l*(M + m - m*cos(X[2])*cos(X[2])))  ))

# Lagrange objective
stage.add_objective(stage.integral(sumsqr(X) + F**2))

# Path constraints
stage.subject_to(F<=10)
stage.subject_to(-10<=F)
stage.subject_to(X[0]>=-0.45)
stage.subject_to(X[0]<=0.45)

# Initial constraints
stage.subject_to(stage.at_t0(X)==X_0)

# Set initial value for the parameters
stage.set_value(X_0, [0, 0, 3.14159265, 0])

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=20,M=1,intg='rk'))

# solve
sol = ocp.solve()

# First solution
ts,thetasol = sol.sample(stage,X[2],grid=stage.grid_control)
print(thetasol)
plt.plot(ts,thetasol,'-o')
