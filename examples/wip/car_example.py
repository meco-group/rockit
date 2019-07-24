from ocpx import *
import matplotlib.pyplot as plt
import numpy as np

# Inspired from https://github.com/casadi/casadi/blob/master/docs/examples/python/direct_multiple_shooting.py

ocp = OcpMultiStage()

stage = ocp.stage(t0=0, T=ocp.free(1.0))  # T initialised at 1, T>=0

# Define constants
m = 500
c = 2
d = 1000
F_max = 2500

# Define states
x1 = stage.state()
x2 = stage.state()

# Defince controls
u = stage.control()

# Specify ODE
stage.set_der(x1, x2)
stage.set_der(x2, 1/m * (u - c * x2**2))

# Lagrange objective
stage.add_objective(stage.T)

# Path constraints
stage.subject_to(u<=F_max)
stage.subject_to(-F_max<=u)
stage.subject_to(x2>=0)

# Initial constraints
stage.subject_to(stage.at_t0(x1)==0)
stage.subject_to(stage.at_t0(x2)==0)

# End constraints
stage.subject_to(stage.at_tf(x1)==d)
stage.subject_to(stage.at_tf(x2)==0)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=20,M=1,intg='rk'))

# solve
sol = ocp.solve()

tsa,x1a = sol.sample(stage,x1,grid=stage.grid_control)
tsa,x2a = sol.sample(stage,x2,grid=stage.grid_control)
tsa,ua = sol.sample(stage,u,grid=stage.grid_control)

tsb,x1b = sol.sample(stage,x1,grid=stage.grid_integrator)
tsb,x2b = sol.sample(stage,x2,grid=stage.grid_integrator)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(tsb,  x1b,'.-')
ax[0].plot(tsa, x1a,'o')
ax[1].plot(tsb,  x2b,'.-')
ax[1].plot(tsa, x2a,'o')
ax[1].legend(['grid_integrator', 'grid_control'])
for i in range(2):
    ax[i].set_xlabel('Time [s]', fontsize=14)
    ax[i].set_ylabel('State {}'.format(i+1), fontsize=14)
plt.show(block=True)

fig, ax = plt.subplots()
ax.plot(tsa, ua,'o')



