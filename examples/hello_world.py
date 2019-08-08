from ocpx import *

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
ocp.subject_to(x1 >= -0.25) # grid='inf')

# Initial constraints
ocp.subject_to(ocp.at_t0(x1) == 0)
ocp.subject_to(ocp.at_t0(x2) == 1)

# Pick an NLP solver backend
ocp.solver('ipopt')

# Pick a solution method
method = MultipleShooting(N=10, M=1, intg='rk')
#method = DirectCollocation(N=20)
ocp.method(method)

# Solve
sol = ocp.solve()

# Show structure
ocp.spy()

# Post-processing
tsa, x1a = sol.sample(x1, grid='control')
tsa, x2a = sol.sample(x2, grid='control')

tsb, x1b = sol.sample(x1, grid='integrator')
tsb, x2b = sol.sample(x2, grid='integrator')


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

try: 
  tsol, usol = sol.sample(u, grid='integrator',refine=100)

  figure()
  plot(tsol,usol)
  title("Control signal")
  xlabel("Times [s]")
  grid(True)

  tsc, x1c = sol.sample(x1, grid='integrator', refine=100)

  figure(figsize=(15, 4))
  plot(tsc, x1c, '-')
  plot(tsa, x1a, 'o')
  plot(tsb, x1b, '.')
  xlabel("Times [s]")
  grid(True)
except:
  pass

show(block=True)
