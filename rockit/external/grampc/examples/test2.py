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

ocp.add_objective(ocp.T)
ocp.add_objective(ocp.at_tf((v-1)**2))

ocp.subject_to(-1<= (u <= 1))

ocp.subject_to(ocp.at_t0(p)==0)
ocp.subject_to(ocp.at_t0(v)==0)

ocp.subject_to(ocp.at_tf(p)==1)

ocp.solver('ipopt',{"ipopt.tol":1e-10})
#ocp.solver('sqpmethod')
#ocp.set_initial(u,1)

#ocp.set_initial(ocp.T, 1.1524561547041494)
#ocp.set_initial(p,[0.     ,    0.    ,     0.10625242 ,0.31875725 ,0.63751449 ,1.00000001])
#ocp.set_initial(v,[0.     ,    0.46098246 ,0.92196492 ,1.38294739 ,1.57266512, 1.11168266])
#ocp.set_initial(u,[ 1.     ,     1.       ,   1.     ,     0.41155087 ,-1.   ,      -1.        ])


# Obtain reference solution
N = 2
ocp.method(MultipleShooting(N=N,M=1,intg="expl_euler"))
sol = ocp.solve()

t,p_sol = sol.sample(p,grid='control')
t,v_sol = sol.sample(v,grid='control')
t,u_sol = sol.sample(u,grid='control')


print(p_sol)
print(v_sol)
print(u_sol)

tf,p_solf = sol.sample(p,grid='integrator',refine=100)
tf,v_solf = sol.sample(v,grid='integrator',refine=100)
tf,u_solf = sol.sample(u,grid='integrator',refine=100)

print(sol.value(ocp.T))

print(sol.value(ocp.T+ocp.at_tf((v-1)**2)))

print("T",sol.value(ocp.T))

print("v",sol.value(ocp.at_tf(v)))

print("obj",sol.value(ocp.T+ocp.at_tf((v-1)**2)))

from casadi import Opti

opti = Opti()

T = opti.variable()
u1 = opti.variable()
u2 = opti.variable()

ut = u1+u2

opti.minimize(ut**2*T**2+(1-2*ut)*T+1)
opti.subject_to(0<= (T<= 100))
opti.subject_to(-1<= (u1<= 1))
opti.subject_to(-1<= (u2<= 1))
opti.subject_to(T/2*u1*T==1)

opti.solver('ipopt')

sol = opti.solve()

print("T:",sol.value(T))
print("u:",sol.value(u1),sol.value(u2))

ocp = Ocp()

T = ocp.variable()
u1 = ocp.variable()
u2 = ocp.variable()

ut = u1+u2

ocp.add_objective(ut**2*T**2+(1-2*ut)*T+1)
ocp.subject_to(0<= (T<= 100))
ocp.subject_to(-1<= (u1<= 1))
ocp.subject_to(-1<= (u2<= 1))
ocp.subject_to(u1*T**2==2)

N = 2
ocp.method(MultipleShooting(N=N,M=1,intg="expl_euler"))
ocp.solver('ipopt')

sol = ocp.solve()

print("T:",sol.value(T))
print("u:",sol.value(u1),sol.value(u2))

ocp = Ocp()

d = ocp.state()
ocp.set_der(d, 0)

T = ocp.variable()
u = ocp.variable()
ut = 1+u
ocp.add_objective((1+u)**2*T**2-2*u*T+1)
ocp.subject_to(0<= (T<= 100))
ocp.subject_to(-1<= (u<= 1))
ocp.subject_to(T**2==2)

N = 2
ocp.method(MultipleShooting(N=N,M=1,intg="expl_euler"))
ocp.solver('ipopt')

sol = ocp.solve()

print("T:",sol.value(T))
print("u:",sol.value(u))

raise Exception()
from pylab import *

plot(t,p_sol,'ro')
plot(tf,p_solf,'r')
plot(t,v_sol,'bo',label='p')
plot(tf,v_solf,'b',label='v')
plot(t,u_sol,'ko')
plot(tf,u_solf,'k',label='u')
legend()
show()