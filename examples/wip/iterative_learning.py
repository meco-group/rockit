from ocpx import MultipleShooting, DirectMethod, OcpMultiStage
from casadi import integrator, vertcat, sin, Function
import numpy as np
import matplotlib.pyplot as plt

def pendulum_ode(x, u, param):
    dx1 = x[1]
    dx2 = (1/(param['I']+param['m']*param['L']**2))*(-param['c']*x[1] - 9.81*param['m']*param['L']*sin(x[0]) + u[0])
    return  vertcat(dx1, dx2)

def pendulum_simulator(x0, u, sim):
    xsim = sim.call({'x0': x0, 'p': u})['xf']
    return {'xsim': xsim, 'ysim': np.squeeze(xsim[0, :])}

#Define problem parameter
plant_param = {'m': 1, 'I': 1, 'L':1, 'c':1}
model_param = {'m': 1.2, 'I': 1.2, 'L':0.9, 'c':0.9}
N = 100
T = 10
tgrid = np.linspace(0, T, N)
yr = np.sin(tgrid)
x0 = [0, 0]

ocp = OcpMultiStage()
stage = ocp.stage(t0=tgrid[0], T=tgrid[-1])

# Define states
x = stage.state(2)

# Define controls
u = stage.control()

# Specify ODE
model_rhs = pendulum_ode(x, u, model_param)
stage.set_der(x, model_rhs)

# Set initial conditions
stage.set_initial(x, x0)

# Set corrected reference yc = y_r - y_i - P(u_i)
yc = stage.parameter(grid='control')

# Set previous control
u_prev = stage.parameter(grid='control')

# Set ILC objective
stage.add_objective(stage.integral((yc-x[0])**2,grid='control')+stage.integral((u-u_prev)**2, grid='control'))

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=N,M=4,intg='cvodes'))

#Define simulator for plant and model
plant_rhs = pendulum_ode(x, u, plant_param)
opts = {}
opts['tf'] = T/N;
ode = {'x': x, 'p': u, 'ode': plant_rhs}
plant_sim = integrator('xkp1', 'cvodes', ode, opts).mapaccum('simulator', N)
ode = {'x': x, 'p': u, 'ode': model_rhs}
model_sim = integrator('xkp1', 'cvodes', ode, opts).mapaccum('simulator', N)

# Run ILC algorithm
u_prev_val = np.zeros(N)
for i in range(10):
    stage.set_value(u_prev, u_prev_val)
    yi = pendulum_simulator(x0, u_prev_val, plant_sim)['ysim']
    y_mod = pendulum_simulator(x0, u_prev_val, model_sim)['ysim']
    stage.set_value(yc, yr - yi + y_mod)
    sol = ocp.solve()
    _, u_prev_val = sol.sample(stage,u, grid='control')


# Plot last result
y_last = pendulum_simulator(x0, u_prev_val, plant_sim)['ysim']
y_last = np.asarray(y_last)

plt.plot(tgrid, yr,'.-')
plt.plot(tgrid, y_last,'-')
plt.legend(['reference', 'last measurement'])
plt.show(block=True)

# uval = np.zeros(N)
# y_plant = pendulum_simulator(x0, uval, plant_sim)['ysim']
# y_mod = pendulum_simulator(x0, uval, model_sim)['ysim']
#
# y_mod = np.asarray(y_mod)
# y_plant = np.asarray(y_plant)
#
# plt.plot(tgrid, y_mod,'-')
# plt.plot(tgrid, y_plant,'-')
# plt.show(block=True)
