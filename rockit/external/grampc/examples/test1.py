# Make available sin, cos, etc
from audioop import ulaw2lin
from numpy import *
# Import the project
from rockit import *

t0 = 0

ocp = Ocp(T=FreeTime(1.0))

p = ocp.state()
v = ocp.state()

u = ocp.control()

ocp.set_der(p,v)
ocp.set_der(v,2*u) # here

ocp.add_objective(ocp.at_tf(ocp.T+(v-1)**2))

ocp.subject_to(-1<= (u <= 1))

ocp.subject_to(ocp.at_t0(p)==0)
ocp.subject_to(ocp.at_t0(v)==0)

ocp.subject_to(ocp.at_tf(p)==1)


ocp.solver('ipopt',{"ipopt.tol":1e-10})
ocp.set_initial(u,1)


# Obtain reference solution # "ruku45"
N = 2
method = external_method('grampc',N=N,grampc_options={"Integrator": "euler","IntegratorAbsTol":1e-9,"MaxMultIter": 8000, "ConvergenceCheck": "on", "MaxGradIter": 1000, "ConstraintsAbsTol": 1e-8, "ConvergenceGradientRelTol": 1e-8,"LineSearchType": "adaptive"})
ocp.method(method)

try:
    sol = ocp.solve()
except Exception as e:
    print(str(e))
    sol = ocp.non_converged_solution
t,p_sol = sol.sample(p,grid='control')
t,v_sol = sol.sample(v,grid='control')
t,u_sol = sol.sample(u,grid='control')


print(p_sol)
print(v_sol)
print(u_sol)

print("T",sol.value(ocp.T))

print("v",sol.value(ocp.at_tf(v)))

print("obj",sol.value(ocp.T+ocp.at_tf((v-1)**2)))

from pylab import *

plot(t,p_sol,'ro-')
plot(t,v_sol,'bo-')
plot(t,u_sol,'ko-')
legend()
show()
