from numpy import *
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

ocp.set_der(x1, v1)
ocp.set_der(theta, dtheta)

denominator = M + m - m*cos(theta)*cos(theta)

ocp.set_der(v1, (-m*l*sin(theta)*dtheta*dtheta + m*g*cos(theta)*sin(theta)+F)/denominator)
ocp.set_der(dtheta, (-m*l*cos(theta)*sin(theta)*dtheta*dtheta + F*cos(theta)+(M+m)*g*sin(theta))/(l*denominator))

ocp.add_objective(ocp.integral(1e3*x1**2+1e3*theta**2+1e-2*v1**2+1e-2*dtheta**2+1e-2**F**2,grid='control'))

ocp.subject_to(ocp.at_t0(x1)==0.0)
ocp.subject_to(ocp.at_t0(theta)==pi)
ocp.subject_to(ocp.at_t0(v1)==0.0)
ocp.subject_to(ocp.at_t0(dtheta)==0.0)

ocp.subject_to(-80 <= (F <= 80))

acados_interface = AcadosInterface(N=20,qp_solver='PARTIAL_CONDENSING_HPIPM',hessian_approx='GAUSS_NEWTON',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=20)

ocp.method(acados_interface)

sol = ocp.solve()

print(sol.sample(x1,grid='control'))


