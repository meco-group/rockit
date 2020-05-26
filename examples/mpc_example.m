%
%     This file is part of rockit.
%
%     rockit -- Rapid Optimal Control Kit
%     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
%
%     Rockit is free software; you can redistribute it and/or
%     modify it under the terms of the GNU Lesser General Public
%     License as published by the Free Software Foundation; either
%     version 3 of the License, or (at your option) any later version.
%
%     Rockit is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%     Lesser General Public License for more details.
%
%     You should have received a copy of the GNU Lesser General Public
%     License along with CasADi; if not, write to the Free Software
%     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
%
%

% Model Predictive Control example
% ================================

import rockit.*

close all

% -------------------------------
% Problem parameters
% -------------------------------
mcart = 0.5;                 % cart mass [kg]
m     = 1;                   % pendulum mass [kg]
L     = 2;                   % pendulum length [m]
g     = 9.81;                % gravitation [m/s^2]

nx    = 4;                   % the system is composed of 4 states
nu    = 1;                   % the system has 1 input
Tf    = 2.0;                 % control horizon [s]
Nhor  = 50;                  % number of control intervals
dt    = Tf/Nhor;             % sample time

current_X = [0.5;0;0;0];  % initial state
final_X   = [0;0;0;0];    % desired terminal state

Nsim  = 200;                 % how much samples to simulate
add_noise = true;            % enable/disable the measurement noise addition in simulation
add_disturbance = true;      % enable/disable the disturbance addition in simulation

% -------------------------------
% Logging variables
% -------------------------------
pos_history     = zeros(1,Nsim+1);
theta_history   = zeros(1,Nsim+1);
F_history       = zeros(1,Nsim);

% -------------------------------
% Set OCP
% -------------------------------
ocp = Ocp('T',Tf);

% Define states
pos    = ocp.state();  % [m]
theta  = ocp.state();  % [rad]
dpos   = ocp.state();  % [m/s]
dtheta = ocp.state();  % [rad/s]

% Defince controls
F = ocp.control(nu, 'order',0);

% Define parameter
X_0 = ocp.parameter(nx);

% Specify ODE
ocp.set_der(pos, dpos);
ocp.set_der(theta, dtheta);
ocp.set_der(dpos, (-m*L*sin(theta)*dtheta*dtheta + m*g*cos(theta)*sin(theta)+F)/(mcart + m - m*cos(theta)*cos(theta)) );
ocp.set_der(dtheta, (-m*L*cos(theta)*sin(theta)*dtheta*dtheta + F*cos(theta)+(mcart+m)*g*sin(theta))/(L*(mcart + m - m*cos(theta)*cos(theta))));

% Lagrange objective
ocp.add_objective(ocp.integral(F*2 + 100*pos^2));

% Path constraints
ocp.subject_to(-2 <= F <= 2  );
ocp.subject_to(-2 <= pos <= 2);

% Initial constraints
X = [pos;theta;dpos;dtheta];
ocp.subject_to(ocp.at_t0(X)==X_0);
ocp.subject_to(ocp.at_tf(X)==final_X);

% Pick a solution method
options = struct;
options.ipopt.print_level = 0;
options.expand = true;
options.print_time = false;
ocp.solver('ipopt',options);

% Make it concrete for this ocp
ocp.method(MultipleShooting('N',Nhor,'M',1,'intg','rk'));

% -------------------------------
% Solve the OCP wrt a parameter value (for the first time)
% -------------------------------
% Set initial value for parameters
ocp.set_value(X_0, current_X);
% Solve
sol = ocp.solve();

% Get discretisd dynamics as CasADi function (hack)
Sim_pendulum_dyn = ocp.discrete_system;

% Log data for post-processing
pos_history(1)   = current_X(1);
theta_history(1) = current_X(2);

% -------------------------------
% Simulate the MPC solving the OCP (with the updated state) several times
% -------------------------------

for i = 1:Nsim
    disp(['timestep ' num2str(i+1) ' of ' num2str(Nsim)]);
    % Get the solution from sol
    [tsa, Fsol] = sol.sample(F, 'grid','control');
    % Simulate dynamics (applying the first control input) and update the current state
    current_X = getfield(Sim_pendulum_dyn('x0',current_X, 'u',Fsol(1), 'T',dt),'xf');
    % Add disturbance at t = 2*Tf
    if add_disturbance
        if i == round(2*Nhor)-1
            disturbance = [0;0;-1e-1;0];
            current_X = current_X + disturbance;
        end
    end
    % Add measurement noise
    if add_noise
        meas_noise = 5e-4*(rand(nx,1)-[1;1;1;1]); % 4x1 vector with values in [-1e-3, 1e-3]
        current_X = current_X + meas_noise;
    end
    % Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X(1:4));
    % Solve the optimization problem
    sol = ocp.solve();
    % Log data for post-processing
    pos_history(i+1)   = full(current_X(1));
    theta_history(i+1) = full(current_X(2));
    F_history(i)       = Fsol(1);
end

% -------------------------------
% Plot the results
% -------------------------------
time_sim = linspace(0, dt*Nsim, Nsim+1);

figure
plot(time_sim, pos_history, 'r-');
hold on
xlabel('Time [s]');
ylabel('Cart position [m]', 'color','r');
plot([2*Tf,2*Tf],[min([pos_history theta_history]),max([pos_history theta_history])], 'k--');
th = text(2*Tf+0.1,0.025,'disturbance applied');

yyaxis right
plot(time_sim, theta_history, 'b-');
ylabel('Pendulum angle [rad]', 'color','b');

set(th,'Rotation',90)

% -------------------------------
% Animate results
% -------------------------------


figure
ax = gca;
hold on
xlabel('X [m]');
ylabel('Y [m]');
for k = 1:Nsim+1
  cart_pos_k      = pos_history(k);
  theta_k         = theta_history(k);
  pendulum_pos_k  = vertcat(horzcat(cart_pos_k,0), vertcat(cart_pos_k-L*sin(theta_k),L*cos(theta_k))');
  color_k     = repmat([0.95*(1-k/(Nsim+1))],1,3);
  plot(ax, pendulum_pos_k(1,1), pendulum_pos_k(1,2), 's', 'markersize',15, 'color',color_k);
  plot(ax, pendulum_pos_k(1:end,1), pendulum_pos_k(1:end,2), '-', 'linewidth',1.5, 'color',color_k);
  plot(ax, pendulum_pos_k(2,1), pendulum_pos_k(2,2), 'o', 'markersize',10, 'color',color_k);
  pause(dt);
end
