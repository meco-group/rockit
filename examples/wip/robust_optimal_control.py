# robust optimal control
from ocpx import *
from casadi import sumsqr, vertcat, sin, cos, vec, diag, horzcat, sqrt
import matplotlib.pyplot as plt
import numpy as np

def robust_control_stages(ocp,delta):
  """
  Returns
  -------
  stage: :obj: `~ocpx.stage.Stage
  x : :obj: `~casadi.MX

  """
  stage = ocp.stage(t0=ocp.free(0), T=ocp.free(1))
  x = stage.state(2)
  u = stage.control()
  der_state = vertcat(x[1],-0.1*(1-x[0]**2 + delta)*x[1] - x[0] + u + delta)
  stage.set_der(x,der_state)
  stage.subject_to(u <= 50)
  stage.subject_to(u >= -50)
  stage.subject_to(x[0] >= -0.25)
  opti = ocp._method.opti
  L = opti.variable()
  stage.add_objective(L)
  stage.subject_to(L>= sumsqr(x[0]-3))
  bound = lambda t: 2 + 0.1*cos(10*t)
  stage.subject_to(x[0] <= bound(stage.t))
  stage.method(MultipleShooting(N=20, M=6, intg='rk'))

  
  return stage,x,stage.at_t0(u)
  
ocp = OcpMultiStage()
ocp.method(DirectMethod(solver='ipopt'))
 
stage1, x1, ut1 = robust_control_stages(ocp,1)
ocp.subject_to(stage1.t0==0)
ocp.subject_to(stage1.tf==1.0)
ocp.subject_to(stage1.at_t0(x1)==(1.0,0))

stage2, x2, ut2 = robust_control_stages(ocp,-1)
ocp.subject_to(stage2.t0==0)
ocp.subject_to(stage2.tf==1.0)
ocp.subject_to(stage2.at_t0(x2)==(1.0,0))
ocp.subject_to(ut1 == ut2)

stage3, x3, ut3 = robust_control_stages(ocp,1)
ocp.subject_to(stage3.t0==1.0)
ocp.subject_to(stage3.tf==2.0)
ocp.subject_to(stage3.at_t0(x3)==stage1.at_tf(x1))

stage4, x4, ut4 = robust_control_stages(ocp,-1)
ocp.subject_to(stage4.t0==1.0)
ocp.subject_to(stage4.tf==2.0)
ocp.subject_to(stage4.at_t0(x4)==stage1.at_tf(x1))
ocp.subject_to(ut3 == ut4)

stage5, x5, ut5 = robust_control_stages(ocp,1)
ocp.subject_to(stage5.t0==1.0)
ocp.subject_to(stage5.tf==2.0)
ocp.subject_to(stage5.at_t0(x5)==stage2.at_tf(x2))

stage6, x6, ut6 = robust_control_stages(ocp,-1)
ocp.subject_to(stage6.t0==1.0)
ocp.subject_to(stage6.tf==2.0)
ocp.subject_to(stage6.at_t0(x6)==stage2.at_tf(x2))
ocp.subject_to(ut5 == ut6)


sol = ocp.solve()

ts1, xsol1 = sol.sample(stage1, x1[0], grid=stage1.grid_control)
ts2, xsol2 = sol.sample(stage2, x2[0], grid=stage2.grid_control)
ts3, xsol3 = sol.sample(stage3, x3[0], grid=stage3.grid_control)
ts4, xsol4 = sol.sample(stage4, x4[0], grid=stage4.grid_control)
ts5, xsol5 = sol.sample(stage5, x5[0], grid=stage5.grid_control)
ts6, xsol6 = sol.sample(stage6, x6[0], grid=stage6.grid_control)

plt.plot(ts1, xsol1, '-o')
plt.plot(ts2, xsol2, '-o')
plt.plot(ts3, xsol3, '-o')
plt.plot(ts4, xsol4, '-o')
plt.plot(ts5, xsol5, '-o')
plt.plot(ts6, xsol6, '-o')
  
bound = lambda t: 2 + 0.1*cos(10*t)

plt.plot(ts1, bound(ts1))
plt.plot(ts3, bound(ts3))

plt.show()