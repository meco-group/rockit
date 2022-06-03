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



from pylab import *

plot(t,p_sol,'ro')
plot(tf,p_solf,'r')
plot(t,v_sol,'bo',label='p')
plot(tf,v_solf,'b',label='v')
plot(t,u_sol,'ko')
plot(tf,u_solf,'k',label='u')
legend()
show()