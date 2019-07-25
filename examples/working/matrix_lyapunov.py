# time optimal example for mass-spring-damper system
from ocpx import *
from casadi import sumsqr, vertcat, sin, cos, vec, diag, horzcat, sqrt, jacobian
import numpy as np

ocp = OcpMultiStage()

stage = ocp.stage(T=1.0)

x = stage.state(2) # two states
u = stage.control()

# nominal dynamics
der_state = vertcat(x[1],-0.1*(1-x[0]**2)*x[1] - x[0] + u)
stage.set_der(x, der_state)

# Lyapunov state
P = stage.state(2, 2)

# Lyapunov dynamics
A = jacobian(der_state, x)
stage.set_der(P, A @ P + P @ A.T)

stage.subject_to(stage.at_t0(x) == vertcat(0.5,0))

P0 = diag([0.01**2,0.1**2])
stage.subject_to(stage.at_t0(P) == P0)
stage.set_initial(P, P0)

stage.subject_to(       u <= 40)
stage.subject_to(-40 <= u      )

stage.subject_to(x[0] >= -0.25)

# 1-sigma bound on x1
sigma = sqrt(horzcat(1,0) @ P @ vertcat(1,0))

# time-dependent bound 
def bound(t):
  return 2 + 0.1*cos(10*t)

# Robustified path constraint
stage.subject_to(x[0] <= bound(stage.t) - sigma)

# Tracking objective
stage.add_objective(stage.integral(sumsqr(x[0]-3)))

ocp.method(DirectMethod(solver='ipopt'))

stage.method(MultipleShooting(N=40, M=6, intg='rk'))

sol = ocp.solve()


# Post-processing

import matplotlib.pyplot as plt

ts, xsol = sol.sample(stage, x[0], grid='control')

plt.plot(ts, xsol, '-o')
plt.plot(ts, bound(ts))

plt.legend(["x1"])
ts, Psol = sol.sample(stage,P,grid = 'control')

o = np.array([[1,0]])
sigma =  np.sqrt(o*Psol*o.T)[:,0,0]
plt.plot([ts,ts],[xsol-sigma,xsol+sigma],'k')

plt.legend(('OCP trajectory x1','bound on x1'))
plt.xlabel('Time [s]')
plt.ylabel('x1')
plt.show()
