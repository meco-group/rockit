ocp = Ocp()

stage = ocp.stage(t0=0,T=10)

x = stage.state()
y = stage.state()
u = stage.control(order=2)

stage.path_constraint(x<=inf)
stage.path_constraint(x>=-0.25)
stage.path_constraint(y<=inf)
stage.path_constraint(y>=-inf)
stage.path_constraint(u<=1)
stage.path_constraint(u>=-1)

stage.add_objective(stage.integral(x**2 + y**2 + u**2)) # Lagrange term
stage.add_objective(stage.at_tf(x**2))                  # Mayer term

stage.set_deriv(x, (1-y**2)*x-y+u+stage.t) # Time-dependant ODE
stage.set_deriv(y, x)

# At time t=0
stage.subject_to(stage.at_t0(x)==0) # Should we automatically recognize path_constraints?
stage.subject_to(stage.at_t0(y)==0)

stage.solver(algo='ms',intg='rk4',N=25,M=4) # 4 integration steps per control

stage.set_initial(x, cos(stage.t))


ocp.solver('ipopt')

sol = ocp.solve()

ts, xsol = sol(stage).sample_intg(x) # Sample at the integrator boundaries (0.1s)
ts, xsol = sol(stage).sample_control(x) # Sample at the control boundaries (0.4s)
ts, xsol = sol(stage).sample_intg_fine(x,N=10) # Refine integration output locally
ts, xsol = sol(stage).sample_sim(x,N=1000) # Simulate feed-forward


