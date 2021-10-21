from pylab import *
from rockit import *
from rockit.acados_interface import AcadosInterface

ocp = Ocp(T=1.0)

M = 1.
m = 0.1
g = 9.81
l = 0.8

x1      = ocp.state()
theta   = ocp.state()
v1      = ocp.state()
dtheta  = ocp.state()

F       = ocp.control()

slbx = ocp.variable(grid='control')
ocp.subject_to(slbx>=0)

subu = ocp.variable(grid='control')
ocp.subject_to(subu>=0)

suh = ocp.variable(grid='control')
ocp.subject_to(suh>=0)

subx = ocp.variable(grid='control')
ocp.subject_to(subx>=0)


p = ocp.parameter()

ocp.set_der(x1, v1)
ocp.set_der(theta, dtheta)

denominator = M + m - m*cos(theta)*cos(theta)

ocp.set_der(v1, (-m*l*sin(theta)*dtheta*dtheta + m*g*cos(theta)*sin(theta)+F)/denominator)
ocp.set_der(dtheta, (-m*l*cos(theta)*sin(theta)*dtheta*dtheta + F*cos(theta)+(M+m)*g*sin(theta))/(l*denominator))

slack = 2*subu+subu**2+slbx+4*subx**2+7*subx+8*slbx**2+3*suh+13*suh**2
ocp.add_objective(ocp.integral(1e3*x1**2+1e3*theta**2+1e-2*v1**2+1e-2*dtheta**2+1e-2**F**2+slack))

ocp.subject_to(ocp.at_t0(x1)==0.0)
ocp.subject_to(ocp.at_t0(theta)==pi)
ocp.subject_to(ocp.at_t0(v1)==0.0)
ocp.subject_to(ocp.at_t0(dtheta)==0.0)

ocp.subject_to(F <= 80+subu)
ocp.subject_to(-10<=theta)
ocp.subject_to(dtheta-subx<=10,include_first=False,include_last=False)
ocp.subject_to(-10<=v1+slbx,include_first=False,include_last=False)

ocp.subject_to(x1+F<=30,include_last=False)
ocp.subject_to(sin(x1)-suh<=30,include_last=False)


#ocp.subject_to(-80 <= (F <= 80+p))

acados_interface = AcadosInterface(N=20,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=20)

ocp.set_value(p, 0)
ocp.method(acados_interface)

sol = ocp.solve()

ts1, x1sol1 = sol.sample(x1,grid='control')

ocp.set_value(p, 3)

sol = ocp.solve()

ts2, x1sol2 = sol.sample(x1,grid='control')

plot(ts1, x1sol1)
plot(ts2, x1sol2)
show()


