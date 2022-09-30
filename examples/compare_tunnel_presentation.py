from casadi import *
from rockit import *
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from numpy import pi, cos, sin, tan, sqrt
import time


#INITIALIZATION
#Initial states
x0 = 0
y0 = 0 
theta0 = pi/2
v0 = 0.1
#Final states 
xf = 0
yf = 10
thetaf = pi/2
vf = 0.1
#Constraints on velocity and acceleration
v_max = 1
a_max = 1
delta_max = pi/6
#Distance between initial and final position (length of the reference path)
L = sqrt((xf-x0)**2+(yf-y0)**2)
#Distance between tires
l = 1
#Number of control intervals
N = 40
M = 2

gammaaa = 1e-8

#Settings for graphs
detail = True
plot_everything = True
dependent_constraints = False #Change constraints graphs

#Solutions are evenly distributed along y direction. Compute the y coordinates
y_co = np.linspace(y0, yf, N+1, endpoint = True)
#Compute the radius of the tunnel for every y coordinate
r = sqrt(1+15*(y_co[:-1]-5)**2)/5

#!OPTIMAL CONTROL PROBLEM Time Domaind = ocpt

ocpt = Ocp(T=FreeTime(10))

# Bicycle model
x     = ocpt.state()
y     = ocpt.state()
theta = ocpt.state()
V     = ocpt.state() #!Add velocity as a state

delta = ocpt.control(order=1)
a = ocpt.control(order=1)   #!Add acceleration as a control

ocpt.set_der(x, V*cos(theta))
ocpt.set_der(y, V*sin(theta))
ocpt.set_der(theta, V/l*tan(delta))
ocpt.set_der(V,a)

# Initial constraints
ocpt.subject_to(ocpt.at_t0(x)==x0)
ocpt.subject_to(ocpt.at_t0(y)==y0)
ocpt.subject_to(ocpt.at_t0(theta)==theta0)
ocpt.subject_to(ocpt.at_t0(V)==v0)

# Final constraint
ocpt.subject_to(ocpt.at_tf(x)==xf)
ocpt.subject_to(ocpt.at_tf(y)==yf)
ocpt.subject_to(ocpt.at_tf(V)==vf)
ocpt.subject_to(ocpt.at_tf(theta)==thetaf)

ocpt.set_initial(delta,0)
ocpt.set_initial(V,0.1) #!v must be different from 0 to avoid division over 0
ocpt.set_initial(theta, pi/2)

#Function that limits the maximum velocity and acceleration depending on delta
lim = Function('lim',[delta],[1/(1+10*delta**2)])

#! Different constraints to see whether the oscillations are due to the constraints
ocpt.subject_to(0 < (V<= v_max), grid = 'integrator')
ocpt.subject_to(-a_max <= (a<= a_max))
ocpt.subject_to( -delta_max<= (delta<= delta_max))

ocpt.subject_to(theta>0) #Set this to avoid division over 0


#ocpt.subject_to(0 < (V<= v_max*lim(delta)))
#ocpt.subject_to(-a_max*lim(delta) <= (a<= a_max*lim(delta)))

ocpt.subject_to( (x+3)**2 <= (sqrt(1+15*(y-5)**2)/5)**2) #The constraint for collision avoidance dependes on another state

# Minimal time
ocpt.add_objective(ocpt.T)

# Pick a solution method
ocpt.solver('ipopt',{"ipopt.max_iter":1000, "expand": True})

# Make it concrete for this ocp
ocpt.method(MultipleShooting(N=N,M=M,intg='rk'))

start = time.time()
# solve
try:
    sol = ocpt.solve()
except:
    sol = ocpt.non_converged_solution
from pylab import *
end = time.time()
comp_time_time = end-start

#Data time solution
T_end_time = sol.value(ocpt.T)

tst, xst = sol.sample(x, grid='control')  # x sampled control grid
tst, yst = sol.sample(y, grid='control')  # y sampled control grid
tsirt, xs_irt = sol.sample(x, grid='integrator',refine=10) #x sampled refined integrator grid
tsirt, ys_irt = sol.sample(y, grid='integrator',refine=10) #y sampled refined integrator grid
tst, vst = sol.sample(V, grid='control')  # velocity sampled control grid
tsirt, vs_irt = sol.sample(V, grid='integrator',refine=10 )  # velocity sampled refined integrator grid
tst, a_st = sol.sample(a, grid='control') # acceleration sampled control grid
tsirtt, a_s_irt = sol.sample(a, grid='integrator',refine=10)   # acceleration sampled refined integrator grid
tst, deltast = sol.sample(delta, grid='control') # delta sampled control grid
tsirt, deltas_irt = sol.sample(delta,  grid='integrator',refine=10) # delta sampled refined integrator grid
tst, thetast = sol.sample(theta, grid='control') # theta sampled control grid

distance_time = sum(sqrt((xst[1:]-xst[:-1])**2 + (yst[1:]-yst[:-1])**2))

##! OPTIMAL CONTROL PROBLEM SPACE
#ocp = Ocp(T=10) # The time horizon is transformed into the length of the reference path #!This line doesn't work. Why?
#ocps = Ocp(T=FreeTime(L))
ocps = Ocp(T=L)

rp = ocps.parameter(grid='control')
s = ocps.t #! s is the independent variable

#Bicycle model
#States
x = ocps.state()
y = ocps.state()
theta = ocps.state()
v = ocps.state() #!Add the velocity as a state
t = ocps.state()
#Control inputs
delta = ocps.control(order=1)
a = ocps.control(order=1) #!Add the acceleration as a control

#Space dynamics
ocps.set_der(x, cos(theta)/(sin(theta)))
ocps.set_der(y, 1)
ocps.set_der(theta, tan(delta)/(l*sin(theta)))
ocps.set_der(v,a/(v*sin(theta)))
ocps.set_der(t, 1/(v*sin(theta)))

ocps.set_initial(delta,0)
ocps.set_initial(v,0.1) #!v must be different from 0 to avoid division over 0
ocps.set_initial(theta, pi/2)

#Boundary conditions
ocps.subject_to(ocps.at_t0(x) == x0)
ocps.subject_to(ocps.at_t0(y) == y0)
ocps.subject_to(ocps.at_t0(theta) == theta0)
ocps.subject_to(ocps.at_t0(v) == v0)
ocps.subject_to(ocps.at_t0(t) == 0)

ocps.subject_to(ocps.at_tf(x) == xf)
ocps.subject_to(ocps.at_tf(y) == yf)
ocps.subject_to(ocps.at_tf(theta)==thetaf)
ocps.subject_to(ocps.at_tf(v) == vf)


#Function that computes the maximum velocity and acceleration depending on delta
lim = Function('lim',[delta],[1/(1+10*delta**2)])

ocps.subject_to( -delta_max <= (delta<= delta_max))
ocps.subject_to(0 < (v<= v_max), grid = 'integrator')
ocps.subject_to(-a_max <= (a<= a_max))

#ocps.subject_to( 0 < (v<= lim(delta)* v_max))
#ocps.subject_to(  -lim(delta)* a_max<= (a<= lim(delta)* a_max))

ocps.subject_to(theta>0) #Set this to avoid division over 0

#! Constraint collision avoidance
#rp = ocps.parameter(grid='control')
ocps.subject_to(-rp <=(x+3 <= rp))
ocps.set_value(rp, r)

#Objective function
T = ocps.integral(1/(v*sin(theta)))
ocps.add_objective(T)
# Pick a solution method

ocps.solver('ipopt',{"ipopt.max_iter":1000, "expand": True})

# Make it concrete for this ocp
ocps.method(MultipleShooting(N=N,M=M,intg='rk'))

#Solve the problem and compute how much time it takes
start = time.time()
sol = ocps.solve()
end = time.time()
comp_time_space = end-start
##Retrieve data
T_end_space = sol.value(T)

ss, instants = sol.sample(t, grid = 'control')  #time sampled control grid
ssir, instants_i = sol.sample(t, grid='integrator', refine = 10) #time sampled refined integrator grid
ss, xs = sol.sample(x, grid='control') #x sampled control grid
ss, ys = sol.sample(y, grid='control') #y sampled control grid
ssir, xs_ir = sol.sample(x, grid='integrator', refine = 10) #x sampled refined integrator grid
ssir, ys_ir = sol.sample(y, grid='integrator', refine = 10) #y sampled refined integratir grid
distance_space = sum(sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)) #Compute distance covered by the robot

if plot_everything:
    ss, deltas = sol.sample(delta, grid='control')  #delta sampled control grid
    ssir, deltas_ir = sol.sample(delta, grid='integrator', refine = 10) #delta sampled refined integrator grid
    ss, vs = sol.sample(v, grid='control')  #velocity sampled control grid
    ssir, vs_ir = sol.sample(v, grid='integrator', refine = 10) #velocity sampled refined integrator grid
    ss, a_s = sol.sample(a, grid='control') #acceleration sampled control grid
    ssir, as_ir = sol.sample(a, grid='integrator', refine = 10) #acceleration sampled refined integrator grid

#Plot the x and y solutions + tunnel shape
figure()
title('TUNNEL N=' +str(N)+' Start: (' + str("{:.2f}".format(x0)) + ',' + str("{:.2f}".format(y0)) + ') End: (' + str("{:.2f}".format(xf)) + ',' + str("{:.2f}".format(yf)) + ')')
text(x0+0.1, y0, 'Time Domain', color = 'b', fontweight = 'bold')
text(x0+0.1, y0-0.3, 'Space Domain', color = 'g', fontweight = 'bold')

#Choose the axis length
if detail:   
    axis([-4, -2, 4 , 6])
else:
    #text(2.5, 9, 'SPACE DOMAIN \nComputation time: ' + str("{:.2f}".format(comp_time_space)) + 's\nDistance covered: '+ str("{:.2f}".format(distance_space)) + 'm\nTime required: '+ str("{:.2f}".format(T_end_space)) + 's' , fontsize=10)
    axis([-8, 5, -1 , 11])
#Plot the tunnel
y_co = np.linspace(y0, yf, 1000) #Points along the path
r = sqrt(1+15*(y_co-5)**2)/5    #Distance between the path and the tunnel walls
plot(-3+r,y_co, 'r-')   
plot(-3-r,y_co, 'r-')
for i in range(len(y_co)):
    hlines(y= y_co[i], xmin=-3-r[i], xmax=-3+r[i], color='r')
angle = np.linspace(0,pi/4,500, endpoint=True)
rf = sqrt(1+15*(yf-5)**2)/5
for i in range(500):
    hlines(y= yf+rf*sin(angle[i]), xmin=-3-rf*cos(angle[i]), xmax=-3+rf*cos(angle[i]), color='r')
    hlines(y= y0-rf*sin(angle[i]), xmin=-3-rf*cos(angle[i]), xmax=-3+rf*cos(angle[i]), color='r')
plot(-3+rf*cos(angle),yf+rf*sin(angle), 'r-')
plot(-3-rf*cos(angle),yf+rf*sin(angle), 'r-')
plot(-3+rf*cos(angle),y0-rf*sin(angle), 'r-')
plot(-3-rf*cos(angle),y0-rf*sin(angle), 'r-')
#Plot x and y coordinates
plot(xst, yst, 'bo',markersize=4)
plot(xs_irt,ys_irt, 'b-', label = 'TIME DOMAIN\nComputation time: ' + str("{:.2f}".format(comp_time_time)) + 's\nDistance covered: '+ str("{:.2f}".format(distance_time)) + 'm\nTime required: '+ str("{:.2f}".format(T_end_time)) + 's')
plot(xs, ys, 'go', markersize=4)
plot(xs_ir,ys_ir, 'g-', label = 'SPACE DOMAIN\nComputation time: ' + str("{:.2f}".format(comp_time_space)) + 's\nDistance covered: '+ str("{:.2f}".format(distance_space)) + 'm\nTime required: '+ str("{:.2f}".format(T_end_space)) + 's')
xlabel('x [m]')
ylabel('y [m]')
legend()
#Plot horizontal lines (to highlight that the solutions are evenly distributed along y coordinate)
for i in range(N+1):
    hlines(y= ys[i], xmin = -20, xmax=xs[i], color='0', linestyle='-', linewidth=0.2)

#Plot steering angle, velocity and acceleration in space grid and time grid
if plot_everything:
    #Steering angle plot ok

    #Acceleration plot
    figure()
    subplot(2,1,1)
    title('Acceleration N=' +str(N))
    if dependent_constraints:
        plot(ss,a_max*1/(1+10*deltas**2), 'r-',linewidth=2.5, label='Bounds') #Plot upper bound on acceleration space
        plot(ss,-a_max*1/(1+10*deltas**2), 'r-', linewidth=2.5)    #Plot lower bound on acceleration space
        plot(yst,a_max*1/(1+10*deltast**2), 'r-',linewidth=2.5) #Plot upper bound on acceleration time
        plot(yst,-a_max*1/(1+10*deltast**2), 'r-', linewidth=2.5)    #Plot lower bound on acceleration time
    else:
        plot(ss,a_max*np.ones(len(ss)), 'r-', linewidth=2.5)   #Plot upper bound on acceleration
        plot(ss,-a_max*np.ones(len(ss)), 'r-', linewidth=2.5)   #Plot lower bound on acceleration

    plot(ss,a_s,'go',markersize=4)   #Control grid space domain
    plot(ssir,as_ir, 'g-', label='SPACE DOMAIN')    #Refined integrator grid space domain
    plot(yst,a_st,'bo',markersize=4) #Control grid time domain
    plot(ys_irt,a_s_irt,'b-', label='TIME DOMAIN')   #Refined integrator grid time domain
    ylabel('Acceleration [m/s^2]', fontsize=14)
    xlabel('Space [m]', fontsize=14)
    axis([0, L+1, -1.25, 1.25])
    legend(loc = 'upper right')
    #!Acceleration plot time grid
    subplot(2,1,2)
    if dependent_constraints:
        plot(instants,a_max*1/(1+10*deltas**2), 'r-', linewidth=2.5)   #Plot upper bound on acceleration space
        plot(instants,-a_max*1/(1+10*deltas**2), 'r-', linewidth=2.5)  #Plot lower bound on acceleration space
        plot(tst,a_max*1/(1+10*deltast**2), 'r-', linewidth=2.5)   #Plot upper bound on acceleration time
        plot(tst,-a_max*1/(1+10*deltast**2), 'r-', linewidth=2.5)  #Plot lower bound on acceleration time
    else:
        plot(instants,a_max*np.ones(len(ss)), 'r-', linewidth=2.5)   #Plot upper bound on acceleration
        plot(instants,-a_max*np.ones(len(ss)), 'r-', linewidth=2.5)   #Plot lower bound on acceleration
    plot(instants,a_s,'go',markersize=4) #Control grid space domain
    plot(instants_i,as_ir, 'g-')  #Integrator grid space domain
    plot(tst,a_st,'bo',markersize=4)  #Control grid time domain 
    plot(tsirt,a_s_irt, 'b-')   #Integrator grid time domain
    ylabel('Acceleration [m/s^2]', fontsize=14)
    xlabel('Time [s]', fontsize=14)
    axis([0, max(T_end_space, T_end_time)+1, -1.25, 1.25])
    figure()
    subplot(2,1,1)
    title('Steering angle delta N='+str(N))
    plot(ss,pi/6*np.ones(len(ss)), 'r-', linewidth=2.5, label='Bounds') #Plot upper bound on delta
    plot(ss, -pi/6*np.ones(len(ss)), 'r-', linewidth=2.5)  #Plot lower bound on delta
    bound = "{:.2f}".format(pi/6) #Upper and lower limit of steering angle
    plot(ss,deltas,'go',markersize=4) #Control grid space domain
    plot(ssir,deltas_ir,'g-', label='SPACE DOMAIN')   #Refined integrator grid space domain
    plot(yst,deltast,'bo', markersize=4) #Control grid time domain
    plot(ys_irt,deltas_irt,'b-', label='TIME DOMAIN')   #Refined integrator grid time domain 
    text(0.2, pi/6 + 0.05, r'Upper bound = pi/6 (' + str(bound) +') rad')
    text(0.2, -pi/6 - 0.1, r'Lower bound = -pi/6 (-' + str(bound) +') rad')
    ylabel('Steering angle delta [rad]', fontsize=14)
    xlabel('Space [m]', fontsize=14)
    axis([0, L+1, -pi/4-0.2 , pi/4])
    legend()
    #!Steering angle plot time grid
    subplot(2,1,2)  
    plot(instants,pi/6*np.ones(len(instants)), 'r-', linewidth=2.5)    #Plot upper bound on steering angle
    plot(instants, -pi/6*np.ones(len(instants)), 'r-', linewidth=2.5)  #Plot lower bound on steering angle
    plot(instants,deltas,'go',markersize=4)  #Control grid space domain
    plot(instants_i,deltas_ir, 'g-')   #Integrator grid space domain
    plot(tst,deltast,'bo',markersize=4)  #Control grid time domain
    plot(tsirt,deltas_irt, 'b-')   #Integrator grid time domain
    text(0.2, pi/6 + 0.05, r'Upper bound = pi/6 (' + str(bound) +') rad')
    text(0.2, -pi/6 - 0.1, r'Lower bound = -pi/6 (-' + str(bound) +') rad')
    ylabel('Steering angle delta [rad]', fontsize=14)
    xlabel('Time [s]', fontsize=14)
    axis([0, max(T_end_space, T_end_time)+1, -pi/4-0.2 , pi/4])

    #Velocity plot
    figure()
    subplot(2,1,1)
    title('Velocity N=' +str(N))

        ##COMPUTE Velocity integral
    delta_ss = ssir[2]-ssir[1]
    delta_tt = ys_irt[2]-ys_irt[1]
    v_media_space = mean(vs_ir)
    v_media_time = mean(vs_irt)
    print('Medium velocity in space domain: ' +str("{:.4f}".format(v_media_space)))
    print('Medium velocity in time domain: '+str("{:.4f}".format(v_media_time)))
    integrand_v_space = 0
    integrand_v_time = 0
    for n in range(len(vs_ir)):
        integrand_v_space = integrand_v_space+ delta_ss*min(v_max,vs_ir[n])
        integrand_v_time =integrand_v_time + delta_ss*min(v_max,vs_irt[n])

    above = 0
    below = 0
    for k in range(3*M*10,37*M*10):
        if vs_irt[k] > v_max:
            above = above + delta_ss*(vs_irt[k]-v_max)
        else:
            below = below + delta_ss*(v_max-vs_irt[k])
    print('Area Above = ' + str(above))
    print('Area Below = ' + str(below))
    text(9.2,1.25,'Area Above = ' + str(above) +'\nArea Below = ' + str(below) )
    plot(ys_irt[3*M*10], vs_irt[3*M*10], 'ro', markersize = 15)
    plot(ys_irt[37*M*10], vs_irt[37*M*10], 'ro', markersize = 15)


    if dependent_constraints:
        plot(ss,v_max*1/(1+10*deltas**2), 'r-',linewidth=2.5, label='Bounds') #Plot upper bound on velocity for space domain
        plot(yst,v_max*1/(1+10*deltast**2), 'r-',linewidth=2.5) #Plot upper bound on velocity for time domain
    else:
        plot(ss,v_max*np.ones(len(ss)), 'r-', linewidth=2.5)   #Plot upper bound on velocity

    plot(ss, np.zeros(len(ss)), 'r-', linewidth=3)   #Plot lower bound on velocity
    plot(ss,vs,'go',markersize=4)    #Control grid space domain
    plot(ssir,vs_ir, 'g-',label='SPACE DOMAIN \nMedium velocity: ' +str("{:.4f}".format(v_media_space)) +'\nMax velocity: ' + str("{:.4f}".format(max(vs_ir))))    #Refined integrator grid space domain
    plot(yst,vst,'bo',markersize=4) #Control grid time domain
    plot(ys_irt,vs_irt,'b-', label='TIME DOMAIN \nMedium velocity: ' +str("{:.4f}".format(v_media_time)) +'\nMax velocity: ' + str("{:.4f}".format(max(vs_irt))))   #Refined integrator grid time domain
    hlines(y= max(vs_ir), xmin = -20, xmax=20, color='0', linestyle='-', linewidth=0.2)
    hlines(y= max(vs_irt), xmin = -20, xmax=20, color='0', linestyle='-', linewidth=0.2)

    for n in range(len(vs_ir)):
        print(vs_irt[n])
    ylabel('Velocity [m/s]', fontsize=14)
    xlabel('Space [m]', fontsize=14)
    axis([0, L+1, -0.5, 1.5])
    legend()
    #!Velocity plot time domain
    subplot(2,1,2)
    plot(instants, np.zeros(len(ss)), 'r-', linewidth=2.5) #Plot lower bound on velocity for space domain
    if dependent_constraints:
        plot(instants,v_max*1/(1+10*deltas**2), 'r-', linewidth=2.5) #Plot upper bound on velocity
        plot(tst,v_max*1/(1+10*deltast**2), 'r-', linewidth=2.5) #Plot upper bound on velocity for time domain
    else:
        plot(tst,v_max*np.ones(len(ss)), 'r-', linewidth=2.5)   #Plot upper bound on velocity
    plot(instants, vs,'go',markersize=4) #Control grid space domain
    plot(instants_i,vs_ir, 'g-')   #Integrator grid space domain
    plot(tst,vst,'bo',markersize=4)  #Control grid time domain
    plot(tsirt,vs_irt, 'b-')    #Integrator grid time domain
    ylabel('Velocity [m/s]', fontsize=14)
    xlabel('Time [s]', fontsize=14)
    axis([0, max(T_end_space, T_end_time)+1, -0.5, 1.5])
    
show(block=True)


