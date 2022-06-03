# Make available sin, cos, etc
from numpy import *
# Import the project
from rockit import *

# Obtain reference solution # "ruku45"
N = 2
method = external_method('grampc',N=N,grampc_options={"Integrator": "euler","IntegratorAbsTol":1e-9,"MaxMultIter": 8000, "ConvergenceCheck": "on", "MaxGradIter": 1000, "ConstraintsAbsTol": 1e-8, "ConvergenceGradientRelTol": 1e-8,"LineSearchType": "adaptive"})

ocp = Ocp()

d = ocp.state()
e = ocp.control()
ocp.subject_to(-1<= (e<= 1))
ocp.set_der(d, e)
ocp.subject_to(ocp.at_t0(d)==0)

ocp.add_objective(ocp.at_tf(d**2))

"""
T = ocp.variable()
u = ocp.variable()
ut = 1+u
ocp.add_objective((1+u)**2*T**2-2*u*T+1)
ocp.subject_to(0<= (T<= 100))
ocp.subject_to(-1<= (u<= 1))
ocp.subject_to(T**2==2)

ocp.set_initial(T,1)
ocp.set_initial(u,-0.3)

"""


ocp.method(method)


sol = ocp.solve()

print("T:",sol.value(T))
print("u:",sol.value(u))

raise Exception()

from pylab import *

plot(t,p_sol,'ro-')
plot(t,v_sol,'bo-')
plot(t,u_sol,'ko-')
legend()
show()
