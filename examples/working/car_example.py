from ocpx import *
import matplotlib.pyplot as plt

ocp = OcpMultiStage()

stage = ocp.stage(T=ocp.free(1.0))

# Define constants
m = 500
c = 2
d = 1000
F_max = 2500

# Define states
p = stage.state()
v = stage.state()

# Defince controls
F = stage.control()

# Specify ODE
stage.set_der(p, v)
stage.set_der(v, 1/m * (F - c * v**2))

# Lagrange objective
stage.add_objective(stage.T)

# Path constraints
stage.subject_to(          F <= F_max)
stage.subject_to(-F_max <= F         )
stage.subject_to(v >= 0)

# Initial constraints
stage.subject_to(stage.at_t0(p)==0)
stage.subject_to(stage.at_t0(v)==0)

# End constraints
stage.subject_to(stage.at_tf(p)==d)
stage.subject_to(stage.at_tf(v)==0)

# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=20,M=1,intg='rk'))

# solve
sol = ocp.solve()


from pylab import *

# Post-processing
tsa, pa = sol.sample(stage, p, grid='control')
tsa, va = sol.sample(stage, v, grid='control')

tsb, pb = sol.sample(stage, p, grid='integrator')
tsb, vb = sol.sample(stage, v, grid='integrator')


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

tsol, usol = sol.sample(stage, F, grid='control')

figure()
step(tsol,usol,where='post')
title("Control signal")
xlabel("Times [s]")
grid(True)

plt.show(block=True)



