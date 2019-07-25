#       control interval
#       v              v
# |             |            |
# |-------------|            |
# |   |    |    |------------|
# |   |    |    |   |    |   |
#   ^    ^   ^    ^   ^    ^
#    integration interval
#
# grid='control'
#     ocp.grid_intg
#     ocp.grid_intg_roots
#     ocp.grid_intg_fine(10)
#     ocp.grid_inf   #-> spline

ocp = Ocp(t0=0, T=10)


x = ocp.state()  # Framehack to get 'x'
y = ocp.state()
u = ocp.control()
u = ocp.control(order=2)

q = ocp.state(7)
dq = ocp.state(7)

# Don't: ocp.sample_intg(x) # Get x1...x_N+1? Still placeholders

ocp.subject_to(x <= inf)  # path constraint, by default on 'control'
ocp.subject_to(x >= -0.25)
ocp.subject_to(y <= inf)
# 10 finer than the integrator grid
ocp.subject_to(y >= -inf, grid=ocp.grid_intg_fine(10))
# for collocation -> constrain on collocation points
ocp.subject_to(u * x <= 1, grid=ocp.grid_intg_roots)
ocp.subject_to(u >= -1)

ocp.add_objective(ocp.integral(x**2 + y**2 + u**2))  # Lagrange term
ocp.add_objective(ocp.sum(x**2 + y**2 + u**2))      # discrete ?
ocp.add_objective(ocp.sum(x**2 + y**2 + u**2,
                  grid=ocp.grid_intg))      # discrete ?
ocp.add_objective(ocp.at_tf(x**2))                  # Mayer term


ocp.add_objective(ocp.sum(ocp.convex_over_nonlinear('norm_2', sin(x)))

# Slack variables
l=ocp.variables(grid='control')

ocp.subject_to(ocp.convex_over_nonlinear('norm_2', sin(x)) <= rho + l)

# Sub-sampling of constraints!!

ocp.set_der(x, (1 - y**2) * x - y + u + ocp.t)  # Time-dependant ODE
ocp.set_der(y, x)


ocp.set_der(q, (pinocchio))

eT=Function('e_T', {"custom_jacobian": ...})

# syntax error if >=x
ocp.subject_to(ocp.custom_jac(e_T(x, u), [x, u], de / dxdu) >= 0)

# Don't do that; (yet)
# ocp.set_intg(x, 'rk4') #?

# accuracy may not be needed for some states

# ocp.set_intg([x,y], 'rk4') #?
# ocp.set_intg(ocp.all_states, 'rk4') #?

ocp.subject_to(ocp.der(ocp.der(x + y**2)) >= 0)  # cache
# Finite differences on u, unless u is order is high enough
ocp.subject_to(ocp.der(u) >= 0)

# At time t=0
# Should we automatically recognize path_constraints? [Armin: yes ]
ocp.subject_to(ocp.at_t0(x) == 0)
ocp.subject_to(ocp.at_t0(y) == 0)  # pointwise_constraint
ocp.subject_to(ocp.at_t0(y) == ocp.at_tf(y))  # periodic



# 4 integration steps per control  # auto-detect solver (QP)?
ocp.solver('ipopt', algo='ms', intg='rk4', N=25, M=4, casadi_options=dict())

# ordering can be preserved

# ms - multiple-shooting
# ss - single-shooting
# dc - direct collocation
#      pseudo-spectral
# spline
# spline-middle



# different integrators for different subsystem; in linear subsystem -> matrix exponential?

ocp.set_initial(x, cos(ocp.t))

ocp.set_initial(x, (ts, xs))

ocp.solver('ipopt')

sol=ocp.solve()


# what to do with controls? drop the last.
# Sample at the control boundaries (0.4s); default
ts, xsol=sol.sample(x, grid='control')
# Sample at the integrator boundaries (0.1s)
ts, xsol=sol.sample(x, grid=ocp.grid_intg)
ts, xsol=sol.sample(x, grid=ocp.grid_intg_roots)  # Collocation grid
# Refine integration output locally
ts, xsol=sol.sample(x, grid=ocp.grid_intg_fine(10))
ts, xsol=sol.sample(x, grid=ts)  # Custom time grid
# Start integration from each control interval
ts, xsol=sol.sample(x, grid=ts, reset_intg=True)
# Start integration from t0 and up to tf, open loop
ts, xsol=sol.sample(x, grid=ts, reset_intg=False)
# If specified, use cvodes integrator
ts, xsol=sol.sample(x, grid=1000, reset_intg=False, tol=1e-9)
