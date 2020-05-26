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


% Example with matrix state
% =========================
% The example uses the Lyapunov differential equation to approximate
% state covariance along the trajectory

% time optimal example for mass-spring-damper system
import rockit.*
ocp = Ocp('T',1.0);

x = ocp.state(2); % two states
u = ocp.control();

% nominal dynamics
der_state = vertcat(x(2),-0.1*(1-x(1)^2)*x(2) - x(1) + u);
ocp.set_der(x, der_state);

% Lyapunov state
P = ocp.state(2, 2);

% Lyapunov dynamics
A = jacobian(der_state, x);
ocp.set_der(P, A * P + P * A');

ocp.subject_to(ocp.at_t0(x) == [0.5;0]);

P0 = diag([0.01^2,0.1^2]);
ocp.subject_to(ocp.at_t0(P) == P0);
ocp.set_initial(P, P0);
ocp.subject_to(-40 <= u <= 40);

ocp.subject_to(x(1) >= -0.25);

% 1-sigma bound on x1
sigma = sqrt([1,0] * P * [1;0]);

% time-dependent bound
bound = @(t) 2 + 0.1*cos(10*t);

% Robustified path constraint
ocp.subject_to(x(1) <= bound(ocp.t) - sigma);

% Tracking objective
ocp.add_objective(ocp.integral(sumsqr(x(1)-3)));

ocp.solver('ipopt');

ocp.method(MultipleShooting('N',40, 'M',6, 'intg','rk'));

sol = ocp.solve();


% Post-processing


[ts, xsol] = sol.sample(x(1), 'grid','control');

figure
hold on
plot(ts, xsol, '-o');
plot(ts, bound(ts));

legend('x1');
[ts, Psol] = sol.sample(P,'grid','control');

o = [1,0];
for k=1:length(ts)
    sigma = sqrt(o*squeeze(Psol(k,:,:))*o');
    plot([ts(k),ts(k)],[xsol(k)-sigma,xsol(k)+sigma],'k');
end



legend('OCP trajectory x1','bound on x1')
xlabel('Time [s]')
ylabel('x1')
