ocp = Ocp()

stage = ocp.stage(t0=0, T=Ocp.Free(1.0))  # T initialised at 1, T>=0

u = stage.control(spline=...)

#v = stage.integral(v)
#p = stage.integral(u)

stage.set_der(v, u)
stage.set_der(p, v)

stage.path_constraint(v[1] <= 2)  # Time scaling

stage.add_objective(stage.T)  # Minimize time

stage.set_initial(p[0], (ts, xs))  # Grid
stage.set_initial(p[1], sin(stage.t))  # Function

stage.path_constraint(sumsqr(p - p0) >= 0.2)  # Obstacle avoidance

stage.subject_to(stage.at_t0(p) == vertcat(0, 0))
stage.subject_to(stage.at_tf(p) == vertcat(10, 10))
