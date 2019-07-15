ocp = Ocp(t0=0,tf=10) # tf or T deltaT or T


x = ocp.state() # Framehack to get 'x'
y = ocp.state()
u = ocp.control()
u = ocp.control(order=2)

ocp.subject_to(x<=inf) # path constraint
ocp.subject_to(x>=-0.25)
ocp.subject_to(y<=inf)
ocp.subject_to(y>=-inf)
ocp.subject_to(u<=1)
ocp.subject_to(u>=-1)

ocp.add_objective(ocp.integral(x**2 + y**2 + u**2)) # Lagrange term
ocp.add_objective(ocp.sum(x**2 + y**2 + u**2))      # discrete ?
ocp.add_objective(ocp.at_tf(x**2))                  # Mayer term

ocp.set_der(x, (1-y**2)*x-y+u+ocp.t) # Time-dependant ODE
ocp.set_der(y, x)

ocp.set_intg(x, 'rk4') #?

ocp.set_intg([x,y], 'rk4') #?
ocp.set_intg(ocp.all_states, 'rk4') #?

ocp.subject_to(ocp.der(ocp.der(x+y**2))>=0)
ocp.subject_to(ocp.der(u)>=0)

#ocp.path_constraint(x.dot == (1-y**2)*x-y+u+ocp.t)

# At time t=0
ocp.subject_to(ocp.at_t0(x)==0) # Should we automatically recognize path_constraints? [Armin: yes ]
ocp.subject_to(ocp.at_t0(y)==0) # pointwise_constraint

ocp.solver(algo='ms',intg='rk4',N=25,M=4) # 4 integration steps per control


# ms
# ss
# dc
# pseudo-spectral
# spline
# spline-middle



# different integrators for different subsystem; in linear subsystem -> matrix exponential?

ocp.set_initial(x, cos(ocp.t))


ocp.set_initial(x, (ts, xs))

ocp.solver('ipopt') # auto-detect solver (QP)?

sol = ocp.solve()


# what to do with controls? drop the last.
ts, xsol = sol.sample_intg(x) # Sample at the integrator boundaries (0.1s)
ts, xsol = sol.sample_control(x) # Sample at the control boundaries (0.4s)
ts, xsol = sol.sample_intg_fine(x,N=10) # Refine integration output locally
ts, xsol = sol.sample_sim(x,N=1000) # Simulate feed-forward

# AUtomatic plot funcitonality -
