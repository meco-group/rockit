# Make available sin, cos, etc
from numpy import *
# Import the project
from rockit import *

# Obtain reference solution # "ruku45"
N = 2
method = external_method('grampc',N=N,grampc_options={"Integrator": "euler","IntegratorAbsTol":1e-9,"MaxMultIter": 8000, "ConvergenceCheck": "on", "MaxGradIter": 1000, "ConstraintsAbsTol": 1e-8, "ConvergenceGradientRelTol": 1e-8,"LineSearchType": "adaptive"})

ocp = Ocp()

d = ocp.state()
ocp.set_der(d, 0)
ocp.subject_to(ocp.at_t0(d)==0)

T = ocp.variable()
u1 = ocp.variable()
u2 = ocp.variable()

ut = u1+u2

ocp.add_objective(ocp.at_tf(ut**2*T**2+(1-2*ut)*T+1))
ocp.subject_to(0<= (T<= 100))
ocp.subject_to(-1<= (u1<= 1))
ocp.subject_to(-1<= (u2<= 1))
ocp.subject_to(u1*T**2==2)

ocp.set_initial(T,1.4142135689078894)
ocp.set_initial(u1,1)
ocp.set_initial(u2,-0.19617223)


ocp.method(method)


try:
    sol = ocp.solve()
except Exception as e:
    print(str(e))
    sol = ocp.non_converged_solution
print("T:",sol.value(T))
print("u:",sol.value(u2))

raise Exception()

from pylab import *

plot(t,p_sol,'ro-')
plot(t,v_sol,'bo-')
plot(t,u_sol,'ko-')
legend()
show()
