# time optimal example for mass-spring-damper system

from ocpx import *
from casadi import sumsqr, vertcat, sin, cos
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ocp = OcpMultiStage()

stage = ocp.stage(t0=0, T=1.0)  # T initialised at 1, T>=0

x = stage.state(2)
u = stage.control()

stage.set_der(x,vertcat(x[1],-0.1*(1-x[0]**2)*x[1] - x[0] + u))

stage.subject_to(stage.at_t0(x) == vertcat(0.5,0))
stage.subject_to(u <= 50)
stage.subject_to(u >= -50)
bound = 2 + 0.1*cos(10*stage.t)
stage.subject_to(x[0] >= -0.25)
stage.subject_to(x[0] <= bound)
stage.add_objective(stage.integral(sumsqr(x[0]-3)))

ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=40, M=6, intg='rk'))
sol = ocp.solve()