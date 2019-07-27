from ocpx import *
import matplotlib.pyplot as plt

ocp = Ocp(T=FreeTime(1.0))

# Define constants
m = 500
c = 2
d = 1000
F_max = 2500

# Define states
p = ocp.state()
v = ocp.state()

# Defince controls
F = ocp.control()

# Specify ODE
ocp.set_der(p, v)
ocp.set_der(v, 1/m * (F - c * v**2))

# Lagrange objective
ocp.add_objective(ocp.T)

# Path constraints
ocp.subject_to(          F <= F_max)
ocp.subject_to(-F_max <= F         )
ocp.subject_to(v >= 0)

# Initial constraints
ocp.subject_to(ocp.at_t0(p)==0)
ocp.subject_to(ocp.at_t0(v)==0)

# End constraints
ocp.subject_to(ocp.at_tf(p)==d)
ocp.subject_to(ocp.at_tf(v)==0)

# Pick a solver
ocp.solver('ipopt')

# Choose a solution method
ocp.method(MultipleShooting(N=20,M=1,intg='rk'))

# solve
sol = ocp.solve()

from pylab import *

# Post-processing
tsa, pa = sol.sample(ocp, p, grid='control')
tsa, va = sol.sample(ocp, v, grid='control')

tsb, pb = sol.sample(ocp, p, grid='integrator')
tsb, vb = sol.sample(ocp, v, grid='integrator')


figure(figsize=(10, 4))
subplot(1, 2, 1)
plot(tsb, pb, '.-')
plot(tsa, pa, 'o')
xlabel("Times [s]", fontsize=14)
grid(True)
title('State p')

subplot(1, 2, 2)
plot(tsb, vb, '.-')
plot(tsa, va, 'o')
legend(['grid_integrator', 'grid_control'])
xlabel("Times [s]", fontsize=14)
title('State v')
grid(True)

tsol, usol = sol.sample(ocp, F, grid='control')

figure()
step(tsol,usol,where='post')
title("Control signal")
xlabel("Times [s]")
grid(True)

plt.show(block=True)



