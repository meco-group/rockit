from ocpx import *
import matplotlib.pyplot as plt

ocp = OcpMultiStage()

stage = ocp.stage(T=10)

# Define 2 states
x1 = stage.state()
x2 = stage.state()

# Define 1 control
u = stage.control()

# Specify ODE
stage.set_der(x1, (1 - x2**2) * x1 - x2 + u)
stage.set_der(x2, x1)

# Lagrange objective
stage.add_objective(stage.integral(x1**2 + x2**2 + u**2))

# Path constraints
stage.subject_to(      u <= 1)
stage.subject_to(-1 <= u     )
stage.subject_to(x1 >= -0.25 )

# Initial constraints
stage.subject_to(stage.at_t0(x1) == 0)
stage.subject_to(stage.at_t0(x2) == 1)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
method = MultipleShooting(N=10, M=1, intg='rk')
method = DirectCollocation(N=20)
stage.method(method)

# solve
sol = ocp.solve()

# Show structure
ocp.spy()


# Post-processing
tsa, x1a = sol.sample(stage, x1, grid='control')
tsa, x2a = sol.sample(stage, x2, grid='control')

tsb, x1b = sol.sample(stage, x1, grid='integrator')
tsb, x2b = sol.sample(stage, x2, grid='integrator')


from pylab import *

figure(figsize=(10, 4))
subplot(1, 2, 1)
plot(tsb, x1b, '.-')
plot(tsa, x1a, 'o')
xlabel("Times [s]", fontsize=14)
grid(True)
title('State x1')

subplot(1, 2, 2)
plot(tsb, x2b, '.-')
plot(tsa, x2a, 'o')
legend(['grid_integrator', 'grid_control'])
xlabel("Times [s]", fontsize=14)
title('State x2')
grid(True)

tsol, usol = sol.sample(stage, u, grid='control')

figure()
step(tsol,usol,where='post')
title("Control signal")
xlabel("Times [s]")
grid(True)


try:
  tsc, x1c = sol.sample(stage, x1, grid='integrator', refine=10)

  figure(figsize=(15, 4))
  plot(tsc, x1c, '.-')
  plot(tsa, x1a, 'o')
  xlabel("Times [s]")
  grid(True)
except:
  pass

plt.show(block=True)
