from ocpx import MultipleShooting, DirectMethod, OcpMultiStage
# import matplotlib.pyplot as plt
import numpy as np

def pendulum_ode(x, u):
    dx1 = x[1]
    dx2 = (1/(I+m*L**2))*(-c*x[1] - g*m*L*sin(x[0]) + u[0])
    return vcat(dx1, dx2)

tgrid = np.linspace(0, 10, 100)

# dyn = {'x': vertcat(x,lam), 'ode': f}
# simulator = integrator('simulator', 'cvodes', dae, {'grid': tgrid, 'output_t0': True})
# sol = simulator(x0 = V_sol[0:4])["xf"]

ocp = OcpMultiStage()
stage = ocp.stage(t0=tgrid[0], T=tgrid[-1])

# Define states
x1 = stage.state()
x2 = stage.state()

# Defince controls
u = stage.control()

# Specify ODE
stage.set_der(x1, (1-x2**2)*x1 - x2 + u)
stage.set_der(x2, x1)

# # Lagrange objective
# stage.add_objective(stage.integral(x1**2 + x2**2 + u**2))
#
# # Path constraints
# stage.subject_to(u <= 1)
# stage.subject_to(-1 <= u)
# stage.subject_to(x1 >= -0.25)
#
# # Initial constraints
# stage.subject_to(stage.at_t0(x1) == 0)
# stage.subject_to(stage.at_t0(x2) == 1)
#
# # Pick a solution method
# ocp.method(DirectMethod(solver='ipopt'))
#
# # Make it concrete for this stage
# stage.method(MultipleShooting(N=20, M=4, intg='cvodes'))
#
# # solve
# sol = ocp.solve()
#
# # solve
# ts, xsol = sol.sample(stage, x1, grid=stage.grid_control)
# plt.plot(ts, xsol, '-o')
# ts, xsol = sol.sample(stage, x2, grid=stage.grid_control)
# plt.plot(ts, xsol, '-o')
#
# #
# # plt.plot(ts,xsol,'-o')
# # ts,xsol = sol.sample(stage,x2,grid=stage.grid_integrator)
# plt.plot(ts, xsol, '-o')
# plt.legend(["x1", "x2"])
#
# plt.show(block=True)
