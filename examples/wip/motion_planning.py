from ocpx import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr

ocp = Ocp(T=FreeTime(10.0))

# Bicycle model

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

delta = ocp.control()
V     = ocp.control()

L = 1

ocp.set_der(x, V*cos(theta))
ocp.set_der(y, V*sin(theta))
ocp.set_der(theta, V/L*tan(delta))
ocp.set_der(theta, V/L*tan(delta))

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(ocp.at_t0(y)==0)
ocp.subject_to(ocp.at_t0(theta)==0)

# Final constraint
ocp.subject_to(ocp.at_tf(x)==0)
ocp.subject_to(ocp.at_tf(y)==10)


ocp.subject_to(V<=1)
ocp.subject_to( -pi/6 <= (delta<= pi/6))

# Round obstacle
p0 = vertcat(1,5)
r0 = 1

p = vertcat(x,y)
ocp.subject_to(sumsqr(p-p0)>=r0**2)

# Minimal time
ocp.add_objective(ocp.T)
ocp.add_objective(ocp.integral(x**2))

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=20,M=4,intg='rk'))

# solve
sol = ocp.solve()

