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
ocp.method(method)

from pylab import *
def plotme(iter, sol):
  figure(figsize=(15, 4))
  for state, label, color in [(x1,'x1','r'),(x2,'x2','g'),(u,'u','b')]:
    tsa, x1a = sol.sample(state, grid='control')
    tsb, x1b = sol.sample(state, grid='integrator')
    tsc, x1c = sol.sample(state, grid='integrator', refine=100)
    plot(tsc, x1c, color+'-',label=label)
    plot(tsa, x1a, color+'o')
    plot(tsb, x1b, color+'.')
  xlabel("Times [s]")
  ylim([-0.3,1.1])
  title('Iteration %d' % iter)
  legend()
  grid(True)

ocp.callback(plotme)

# Solve
sol = ocp.solve()

show(block=True)