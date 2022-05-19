#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

"""
Time-optimal pathfollowing
======================================

Two-link system see two_link.png
There is a geometric path given yd(s)
We seek to traverse the path exactly, as fast as possible (torque limited)
We start from a certain speed

Transformation given on p13-14 in "Optimal Robot Path Following", PhD thesis, Frederik Debrouwere https://lirias.kuleuven.be/retrieve/336410

"""

from casadi import *
from rockit import *

#Define kinematic parameters
l1 = 1 #length
l2 = 1
m1 = 1 #mass
m2 = 2
I1 = 0.5 #inertia around COG
I2 = 0.5
lc1 = 0.5#COG
lc2 = 0.5

grav = 9.81 #gravitational constant

#Define dynamic parameters
mu11 = m1
mu21 = m1*lc1
mu31 = m1*lc1**2+I1

mu12 = m2
mu22 = m2*lc2
mu32 = m2*lc2**2+I2

#Torque limits
τ_lim = vertcat(30,15)

#Define matrices describing robot dynamics as functions
M = lambda q: blockcat([[mu12*l1**2+mu22*2*l1*cos(q[1])+mu31+mu32,mu32+mu22*l1*cos(q[1])],
          [mu32+mu22*l1*cos(q[1]),mu32]])
      
C = lambda q,q̇: blockcat([[-mu22*l1*sin(q[1])*q̇[1],-mu22*l1*sin(q[1])*(q̇[0]+q̇[1])],
            [mu22*l1*sin(q[1])*q̇[0],0]])
        
G = lambda q: grav*vertcat(mu12*l1*cos(q[0])+mu21*cos(q[0])+mu22*cos(q[0]+q[1]),mu22*cos(q[0]+q[1]))

#Define matrix describing robot kinematics
chi = lambda q: vertcat(l1*cos(q[0])+l2*cos(q[0]+q[1]),l1*sin(q[0])+l2*sin(q[0]+q[1]))

# Inverse kinematics: chi_inv(chi(q0)) == q0
def chi_inv(y):
    c = (y[0]**2+y[1]**2-l1**2-l2**2)/(2*l1*l2)
    q2 = acos(c)
    q1 = atan2(y[1],y[0])-atan2(l2*sin(q2),l1+l2*c)
    return vertcat(q1,q2)

#Define desired trajectory of the end-effector in Cartesian space
L = 1 # s in [0,L]
yd = lambda s : vertcat(1.5,0)*s/L+vertcat(0,1.5)*(1-s/L) #straight line as a function of s

qd = lambda s: chi_inv(yd(s))

T0 = 1
# Start speed
v0 = 1 # [m/s]
ocp = Ocp(T=FreeTime(1))

# Minimize time
ocp.add_objective(ocp.T)

s = ocp.state()
ṡ = ocp.state()
s̈ = ocp.control()
ocp.set_der(s,ṡ)
ocp.set_der(ṡ,s̈)

# No going back along the path
ocp.subject_to(ṡ>=0)

q = qd(s)
q̇ = jtimes(q,s,ṡ)
q̈ = jtimes(q̇,vertcat(s,ṡ),vertcat(ṡ,s̈))

τ = M(q) @ q̈ + C(q,q̇) @ q̇ + G(q)

# Torque constraints
ocp.subject_to(-τ_lim <= (τ <= τ_lim))

# We need to traverse [0,L]
ocp.subject_to(ocp.at_t0(s)==0)
ocp.subject_to(ocp.at_tf(s)==L)

v_squared = sumsqr(ocp.der(yd(s)))

# Start from a given initial speed
ocp.subject_to(ocp.at_t0(v_squared)==v0**2)

# Invent some initial guesses
ocp.set_initial(s, ocp.t*L/T0)
ocp.set_initial(ṡ, L/T0)

ocp.method(MultipleShooting(N=100))

ocp.solver("ipopt",{"expand":True})

sol = ocp.solve()

print("Transit time [s]: ", sol.value(ocp.T))

# N=100:  Transit time [s]:  0.8284092528448338 (18 iters, 0.045s)
# N=1000: Transit time [s]:  0.8329264935973172 (22 iters, 0.539s)
# N=5000: Transit time [s]:  0.8335742877153355 (25 iters, 3.71s)

# Post processing

import matplotlib.pyplot as plt
plt.figure()

plt.subplot(3,1,1)
plt.plot(*sol.sample(s,grid='integrator',refine=10))
plt.title("Progress variable")
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(*sol.sample(ṡ,grid='integrator',refine=10))
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(*sol.sample(s̈,grid='integrator',refine=10))
plt.grid(True)
plt.xlabel("Time [s]")



plt.figure()
plt.plot(*sol.sample(τ,grid='integrator',refine=10))
plt.grid(True)
plt.xlabel("Time [s]")
plt.title("control effort (torques)")


plt.figure()
plt.plot(*sol.sample(q,grid='integrator',refine=10))
plt.grid(True)
plt.xlabel("Time [s]")
plt.title("joint evolution")

plt.figure()


[_,chi_sol] = sol.sample(chi(q),grid='control')
[_,chi_sol_fine] = sol.sample(chi(q),grid='control')
plt.plot(chi_sol[:,0],chi_sol[:,1],'bo')
plt.plot(chi_sol_fine[:,0],chi_sol[:,1],'b')

plt.xlabel("x position [m]")
plt.ylabel("y position [m]")

plt.title("Geometric plot")


plt.show(block=True)


