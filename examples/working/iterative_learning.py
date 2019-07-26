from ocpx import MultipleShooting, DirectMethod, OcpMultiStage
from casadi import integrator, vertcat, sin, Function, vcat, norm_2, horzcat
import numpy as np
import matplotlib.pyplot as plt

def pendulum_ode(x, u, param):
    dx1 = x[1]
    dx2 = (1/(param['I']+param['m']*param['L']**2))*(-param['c']*x[1] - 9.81*param['m']*param['L']*sin(x[0]) + u[0])
    return  vertcat(dx1, dx2)

def pendulum_simulator(x0, u, sim):
    xsim = horzcat(x0, sim.call({'x0': x0, 'p': u})['xf'])
    return {'xsim': xsim, 'ysim': np.squeeze(xsim[0, :-1])}

#Define problem parameter
plant_param = {'m': 1,      'I': 1,     'L':1,      'c':1}
model_param = {'m': 1.2,    'I': 1.2,   'L':0.9,    'c':0.9}
N = 100
T = 10
tgrid = np.linspace(0, T, N)
y_ref_val = np.sin(tgrid)
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
stage.subject_to(stage.at_t0(x) == x0)

# Define reference
y_ref = stage.parameter(grid='control')
stage.set_value(y_ref, y_ref_val)

# Define output correction
beta = stage.parameter(grid='control')

# Define previous control
u_prev = stage.parameter(grid='control')


# Set ILC objective
stage.add_objective(stage.integral((y_ref - beta -x[0])**2,grid='control')+1e-3*stage.integral((u-u_prev)**2, grid='control'))

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=N,M=4,intg='rk'))

# Define simulator for plant and model
plant_rhs = pendulum_ode(x, u, plant_param)

opts = {'tf': T/N}

data = {'x': x, 'p': u, 'ode': plant_rhs}
plant_sim = integrator('xkp1', 'cvodes', data, opts).mapaccum('simulator', N)

data = {'x': x, 'p': u, 'ode': model_rhs}
model_sim = integrator('xkp1', 'cvodes', data, opts).mapaccum('simulator', N)

# Run ILC algorithm
u_prev_val = np.zeros(N)

y_meas = []

num_its = 10

for i in range(num_its):
    # Compute correction
    y_meas.append(pendulum_simulator(x0, u_prev_val, plant_sim)['ysim'])
    y_model = pendulum_simulator(x0, u_prev_val, model_sim)['ysim']
    beta_val = y_meas[-1] - y_model

    # Set parameters for the current ILC iteration
    stage.set_value(u_prev, u_prev_val)
    stage.set_value(beta, beta_val)

    # Solve ILC problem
    sol = ocp.solve()
    t, u_prev_val = sol.sample(stage, u.T, grid='control')
    t, x_val= sol.sample(stage, x, grid='control')
    u_prev_val = u_prev_val[:-1]


    # plt.plot(tgrid, u_prev_val,'.-')
    # plt.plot(tgrid, y_ref_val,'.-')
    # plt.plot(tgrid, x_val[:-1,0],'-')
    # plt.legend(['u', 'pos ref ','position'])
    # plt.show(block=True)

# Plot last result
y_meas.append(pendulum_simulator(x0, u_prev_val, plant_sim)['ysim'])
y_first = np.asarray(y_meas[1])
y_last = np.asarray(y_meas[-1])

plt.figure()
plt.plot(tgrid, y_ref_val,'.-')
plt.plot(tgrid, y_first,'-')
plt.plot(tgrid, y_last,'-')
plt.legend(['reference', 'first measurement','last measurement'])
# plt.show(block=True)

err_norm_evol = np.zeros([num_its+1,1])
for i in range(num_its+1):
    err_norm_evol[i] = norm_2(y_ref_val - y_meas[i])

plt.figure()
plt.plot(range(num_its+1), err_norm_evol,'-')
plt.legend(['error norm'])
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
