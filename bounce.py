
ocp = OcpMultiStage()

stage = ocp.stage() # Omitting means variable

p = stage.state() # Framehack to get stage['s']
v = stage.state()

stage.set_der(p,v)
stage.set_der(v,-9.81)

stage.path_constraint(p<=5)

#stage.subject_to(stage.t0==0)
#ocp.subject_to(stage.t0==0)

ocp.subject_to(stage.t0==0) # not stage.subject_to !

s = stage

stages = [s]
for i in range(10):
    ocp.subject_to(s.at_t0(p)==0)
    ocp.subject_to(s.at_tf(p)==0)

    s_next = ocp.stage(stage) # copy constructor
    #s.subject_to(stage.t0==0)
    
    ocp.subject_to(s.tf==s_next.t0)
    ocp.subject_to(s.at_tf(v)==-0.9*s_next.at_t0(v)) # Bouncing inverts (and diminimishes velocity)

    sk = s_next
    stages.append(s_next)

sol = ocp.solve()

for s in stages:
  ts, xs = sol(s).sample_sim(x)
  plot(ts, xs)















