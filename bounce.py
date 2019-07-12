
ocp = Ocp()

stage = ocp.stage() # Omitting means variable

s = stage.state() # Framehack to get stage['s']
v = stage.state()

stage.set_der(s,v)
stage.set_der(v,-10)

stage.path_constraint(s<=5)

ocp.subject_to(stage.t0==0) # not stage.subject_to !

sk = stage

stages = [sk]
for i in range(10):
    ocp.add(sk)
    ocp.subject_to(sk.at_t0(s)==0)
    ocp.subject_to(sk.at_tf(s)==0)
    ocp.subject_to(sk.at_tf(v)==-0.9*sk.at_t0(v)) # Bouncing inverts (and diminimishes velocity)
    ocp.subject_to(sk.tf==skp.t0)

    sk, skp = skp, ocp.stage(stage) # copy constructor

    stages.append(skp)

sol = ocp.solve()

for s in stages:
  ts, xs = sol(s).sample_sim(x)
  plot(ts, xs)















