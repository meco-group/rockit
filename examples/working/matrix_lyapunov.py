# time optimal example for mass-spring-damper system

from ocpx import *
from casadi import sumsqr, vertcat, sin, cos, vec, diag
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ocp = OcpMultiStage()

stage = ocp.stage(t0=0, T=1.0)  # T initialised at 1, T>=0

x = stage.state(2)
u = stage.control()
P = stage.state(2,2)
der_state = vertcat(x[1],-0.1*(1-x[0]**2)*x[1] - x[0] + u)
A = stage.get_jacobian(der_state,x)
stage.set_der(x,der_state)
stage.set_der(P,A@P+P@A.T)
#import ipdb
#ipdb.set_trace()

stage.subject_to(stage.at_t0(x) == vertcat(0.5,0))
stage.subject_to(stage.at_t0(P)==diag([0.01**2,0.1**2]))
stage.subject_to(u <= 40)
stage.subject_to(u >= -40)
bound = lambda t: 2 + 0.1*cos(10*t)
stage.subject_to(x[0] >= -0.25)
stage.subject_to(x[0] <= bound(stage.t))
stage.add_objective(stage.integral(sumsqr(x[0]-3)))

ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=40, M=6, intg='rk'))
sol = ocp.solve()

ts, xsol = sol.sample(stage, x[0], grid='control')

plt.plot(ts, bound(ts), '-o')
#
# plt.plot(ts,xsol,'-o')
#ts,xsol = sol.sample(stage,x2,grid='integrator')
plt.legend(["x1"])

plt.show()