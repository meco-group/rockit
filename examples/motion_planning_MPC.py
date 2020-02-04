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
Motion planning model predictive control scheme
===============================================

"""

from rockit import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr



# -------------------------------
# Define some functions to match current position with reference path
# -------------------------------

# Find closest point on the reference path compared witch current position
def find_closest_point(pose, reference_path, start_index):
    # x and y distance from current position (pose) to every point in 
    # the reference path starting at a certain starting index
    xlist = reference_path['x'][start_index:] - pose[0]
    ylist = reference_path['y'][start_index:] - pose[1]
    # Index of closest point by Pythagoras theorem
    index_closest = start_index+np.argmin(np.sqrt(xlist*xlist + ylist*ylist))
    print('find_closest_point results in', index_closest)
    return index_closest

# Return the point on the reference path that is located at a certain distance 
# from the current position
def index_last_point_fun(start_index, wp, dist):
    pathpoints = wp.shape[1]
    # Cumulative distance covered
    cum_dist = 0
    # Start looping the index from start_index to end
    for i in range(start_index, pathpoints-1):
        # Update comulative distance covered
        cum_dist += np.linalg.norm(wp[:,i] - wp[:,i+1])
        # Are we there yet?
        if cum_dist >= dist:
            return i + 1
    # Desired distance was never covered, -1 for zero-based index
    return pathpoints - 1

# Create a list of N waypoints
def get_current_waypoints(start_index, wp, N, dist):
    # Determine index at reference path that is dist away from starting point
    last_index = index_last_point_fun(start_index, wp, dist)
    # Calculate amount of indices between last and start point
    delta_index = last_index - start_index
    # Dependent on the amount of indices, do
    if delta_index >= N: 
        # There are more than N path points available, so take the first N ones
        index_list = list(range(start_index, start_index+N+1))
        print('index list with >= N points:', index_list)
    else:
        # There are less than N path points available, so add the final one multiple times
        index_list = list(range(start_index, last_index)) + [last_index]*(N-delta_index+1)
        print('index list with < N points:', index_list)
    return wp[:,index_list]


# -------------------------------
# Problem parameters
# -------------------------------

Nsim    = 30            # how much samples to simulate
L       = 1             # bicycle model length
nx      = 3             # the system is composed of 3 states
nu      = 2             # the system has 2 control inputs
N       = 10            # number of control intervals

# -------------------------------
# Logging variables
# -------------------------------

time_hist      = np.zeros((Nsim+1, N+1))
x_hist         = np.zeros((Nsim+1, N+1))
y_hist         = np.zeros((Nsim+1, N+1))
theta_hist     = np.zeros((Nsim+1, N+1))
delta_hist     = np.zeros((Nsim+1, N+1))
V_hist         = np.zeros((Nsim+1, N+1))

tracking_error = np.zeros((Nsim+1, 1))

# -------------------------------
# Set OCP
# -------------------------------

ocp = Ocp(T=FreeTime(10.0))

# Bicycle model

# Define states
x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

# Defince controls
delta = ocp.control()
V     = ocp.control(order=0)

# Specify ODE
ocp.set_der(x,      V*cos(theta))
ocp.set_der(y,      V*sin(theta))
ocp.set_der(theta,  V/L*tan(delta))

# Define parameter
X_0 = ocp.parameter(nx)

# Initial constraints
X = vertcat(x, y, theta)
ocp.subject_to(ocp.at_t0(X) == X_0)

# Initial guess
ocp.set_initial(x,      0)
ocp.set_initial(y,      0)
ocp.set_initial(theta,  0)

ocp.set_initial(V,    0.5)

# Path constraints
ocp.subject_to( 0 <= (V <= 1) )
#ocp.subject_to( -0.3 <= (ocp.der(V) <= 0.3) )
ocp.subject_to( -pi/6 <= (delta <= pi/6) )

# Minimal time
# ocp.add_objective(0.50*ocp.T)

# Define physical path parameter
waypoints = ocp.parameter(2, grid='control')
waypoint_last = ocp.parameter(2)
p = vertcat(x,y)

# waypoints = ocp.parameter(3, grid='control')
# waypoint_last = ocp.parameter(3)
# p = vertcat(x,y,theta)

ocp.add_objective(ocp.sum(sumsqr(p-waypoints), grid='control'))
ocp.add_objective(sumsqr(ocp.at_tf(p)-waypoint_last))

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt', options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=2)))

# Define reference path
pathpoints = 30
ref_path = {}
ref_path['x'] = 5*sin(np.linspace(0,2*pi, pathpoints+1))
ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
wp = horzcat(ref_path['x'], ref_path['y']).T

# theta_path = [arctan2(ref_path['y'][k+1]-ref_path['y'][k], ref_path['x'][k+1]-ref_path['x'][k]) for k in range(pathpoints)] 
# ref_path['theta'] = theta_path + [theta_path[-1]]
# wp = horzcat(ref_path['x'], ref_path['y'], ref_path['theta']).T

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------

# First waypoint is current position
index_closest_point = 0

# Create a list of N waypoints
current_waypoints = get_current_waypoints(index_closest_point, wp, N, dist=6)

# Set initial value for waypoint parameters
ocp.set_value(waypoints,current_waypoints[:,:-1])
ocp.set_value(waypoint_last,current_waypoints[:,-1])

# Set initial value for states
current_X = vertcat(ref_path['x'][0], ref_path['y'][0], 0)
ocp.set_value(X_0, current_X)

# Solve the optimization problem
sol = ocp.solve()

# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing  
t_sol, x_sol     = sol.sample(x,     grid='control')
t_sol, y_sol     = sol.sample(y,     grid='control')
t_sol, theta_sol = sol.sample(theta, grid='control')
t_sol, delta_sol = sol.sample(delta, grid='control')
t_sol, V_sol     = sol.sample(V,     grid='control')

time_hist[0,:]    = t_sol
x_hist[0,:]       = x_sol
y_hist[0,:]       = y_sol
theta_hist[0,:]   = theta_sol
delta_hist[0,:]   = delta_sol
V_hist[0,:]       = V_sol

tracking_error[0] = sol.value(ocp.objective)

# Look at the Constrain Jacobian and the Lagrange Hessian structure
ocp.spy()

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    
    # Combine first control inputs
    current_U = vertcat(delta_sol[0], V_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)

    # Find closest point on the reference path compared witch current position
    index_closest_point = find_closest_point(current_X[:2], ref_path, index_closest_point)

    # Create a list of N waypoints
    current_waypoints = get_current_waypoints(index_closest_point, wp, N, dist=6)

    # Set initial value for waypoint parameters
    ocp.set_value(waypoints, current_waypoints[:,:-1])
    ocp.set_value(waypoint_last, current_waypoints[:,-1])

    # Solve the optimization problem
    sol = ocp.solve()

    # Log data for post-processing  
    t_sol, x_sol     = sol.sample(x,     grid='control')
    t_sol, y_sol     = sol.sample(y,     grid='control')
    t_sol, theta_sol = sol.sample(theta, grid='control')
    t_sol, delta_sol = sol.sample(delta, grid='control')
    t_sol, V_sol     = sol.sample(V,     grid='control')
    
    time_hist[i+1,:]    = t_sol
    x_hist[i+1,:]       = x_sol
    y_hist[i+1,:]       = y_sol
    theta_hist[i+1,:]   = theta_sol
    delta_hist[i+1,:]   = delta_sol
    V_hist[i+1,:]       = V_sol
    
    tracking_error[i+1] = sol.value(ocp.objective)
    print('Tracking error f', tracking_error[i+1])

    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(theta, theta_sol)
    ocp.set_initial(delta, delta_sol)
    ocp.set_initial(V, V_sol)

# -------------------------------
# Plot the results
# -------------------------------

T_start = 0
T_end = sum(time_hist[k,1] - time_hist[k,0] for k in range(Nsim+1))

fig = plt.figure()

ax2 = plt.subplot(2, 2, 1)
ax3 = plt.subplot(2, 2, 2)
ax4 = plt.subplot(2, 2, 3)
ax5 = plt.subplot(2, 2, 4)

ax2.plot(wp[0,:], wp[1,:], 'ko')
ax2.set_xlabel('x pos [m]')
ax2.set_ylabel('y pos [m]')
ax2.set_aspect('equal', 'box')

ax3.set_xlabel('T [s]')
ax3.set_ylabel('pos [m]')
ax3.set_xlim(0,T_end)

ax4.axhline(y= pi/6, color='r')
ax4.axhline(y=-pi/6, color='r')
ax4.set_xlabel('T [s]')
ax4.set_ylabel('delta [rad/s]')
ax4.set_xlim(0,T_end)

ax5.axhline(y=0, color='r')
ax5.axhline(y=1, color='r')
ax5.set_xlabel('T [s]')
ax5.set_ylabel('V [m/s]')
ax5.set_xlim(0,T_end)

# fig2 = plt.figure()
# ax6 = plt.subplot(1,1,1)

for k in range(Nsim+1):
    # ax6.plot(time_hist[k,:], delta_hist[k,:])
    # ax6.axhline(y= pi/6, color='r')
    # ax6.axhline(y=-pi/6, color='r')
    
    ax2.plot(x_hist[k,:], y_hist[k,:], 'b-')
    ax2.plot(x_hist[k,:], y_hist[k,:], 'g.')
    ax2.plot(x_hist[k,0], y_hist[k,0], 'ro', markersize = 10)
    
    ax3.plot(T_start, x_hist[k,0], 'b.')
    ax3.plot(T_start, y_hist[k,0], 'r.')

    ax4.plot(T_start, delta_hist[k,0], 'b.')
    ax5.plot(T_start, V_hist[k,0],     'b.')
    
    T_start = T_start + (time_hist[k,1] - time_hist[k,0])
    plt.pause(0.5)

ax3.legend(['x pos [m]','y pos [m]'])

fig3 = plt.figure()
ax1 = plt.subplot(1,1,1)
ax1.semilogy(tracking_error)
ax1.set_xlabel('N [-]')
ax1.set_ylabel('obj f')

plt.show(block=True)
