from rockit import *
import matplotlib.pyplot as plt
import numpy as np

# Inspired from https://github.com/casadi/casadi/blob/master/docs/examples/python/direct_multiple_shooting.py

ocp = Ocp()

stage = ocp.stage(t0=0,T=10)

# Define states
x1 = stage.state()
x2 = stage.state()

# Defince controls
u = stage.control()

p = stage.parameter(grid = 'control')

# Specify ODE
stage.set_der(x1, (1-x2**2)*x1 - x2 + u)
stage.set_der(x2,  x1)

# Lagrange objective
stage.add_objective(stage.integral(x1**2 + x2**2 + u**2))

# Path constraints
stage.subject_to(u<=1)
stage.subject_to(-1<=u)
stage.subject_to(x1>=p)
stage.set_value(p, np.linspace(0,-0.25,20))
# Initial constraints
stage.subject_to(stage.at_t0(x1)==0)
stage.subject_to(stage.at_t0(x2)==1)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=20,M=4,intg='rk'))

# solve
sol = ocp.solve()

# Show structure
ocp.spy()

tsa,x1a = sol.sample(stage,x1,grid='control')
tsa,x2a = sol.sample(stage,x2,grid='control')

tsb,x1b = sol.sample(stage,x1,grid='integrator')
tsb,x2b = sol.sample(stage,x2,grid='integrator')

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
