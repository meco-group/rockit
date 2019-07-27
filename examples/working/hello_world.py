from ocpx import *
import matplotlib.pyplot as plt

ocp = Ocp(T=10)

# Define 2 states
x1 = ocp.state()
x2 = ocp.state()

# Define 1 control
u = ocp.control(order=0)

# Specify ODE
ocp.set_der(x1, (1 - x2**2) * x1 - x2 + u)
ocp.set_der(x2, x1)

# Lagrange objective
ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))

# Path constraints
ocp.subject_to(      u <= 1)
ocp.subject_to(-1 <= u     )
ocp.subject_to(x1 >= -0.25 )

# Initial constraints
ocp.subject_to(ocp.at_t0(x1) == 0)
ocp.subject_to(ocp.at_t0(x2) == 1)

ocp.solver('ipopt')

# Pick a solution method
method = MultipleShooting(N=10, M=1, intg='rk')
#method = DirectCollocation(N=20)
ocp.method(method)

# solve
sol = ocp.solve()

# Show structure
ocp.spy()

# Post-processing
tsa, x1a = sol.sample(ocp, x1, grid='control')
tsa, x2a = sol.sample(ocp, x2, grid='control')

tsb, x1b = sol.sample(ocp, x1, grid='integrator')
tsb, x2b = sol.sample(ocp, x2, grid='integrator')


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

tsol, usol = sol.sample(ocp, u, grid='integrator',refine=100)

figure()
plot(tsol,usol)
title("Control signal")
xlabel("Times [s]")
grid(True)

try:
  tsc, x1c = sol.sample(ocp, x1, grid='integrator', refine=10)

  figure(figsize=(15, 4))
  plot(tsc, x1c, '.-')
  plot(tsa, x1a, 'o')
  xlabel("Times [s]")
  grid(True)
except:
  pass

plt.show(block=True)
