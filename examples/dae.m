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

% Dae example
% ===========
% Example of adding an algebraic state and algebraic equation
import rockit.*

ocp = Ocp('T',10);

% Define 2 states
x1 = ocp.state();
x2 = ocp.state();

% Define 1 control
u = ocp.control('order',0);

z = ocp.algebraic();

% Specify ODE
ocp.set_der(x1, z * x1 - x2 + u);
ocp.set_der(x2, x1);
ocp.add_alg(z-(1 - x2^2));

% Lagrange objective
ocp.add_objective(ocp.integral(x1^2 + x2^2 + u^2));

% Path constraints
ocp.subject_to(-1 <= u <= 1);
ocp.subject_to(-0.25 <= x1 <= 1); % 'grid','inf')

% Initial constraints
ocp.subject_to(ocp.at_t0(x1) == 0);
ocp.subject_to(ocp.at_t0(x2) == 1);

%ocp.set_initial(z, 3)

% Pick an NLP solver backend
ocp.solver('ipopt');

% Pick a solution method
%method = MultipleShooting(N=10, M=1, intg='rk')
method = DirectCollocation('N',10, 'M',2);
ocp.method(method);

% Solve
sol = ocp.solve();

% Show structure
ocp.spy();

% Post-processing
[tsa,x1a] = sol.sample(x1, 'grid','control');
[tsa,x2a] = sol.sample(x2, 'grid','control');

[tsb,x1b] = sol.sample(x1, 'grid','integrator');
[tsb,x2b] = sol.sample(x2, 'grid','integrator');



figure();
subplot(1, 2, 1);
hold on
plot(tsb, x1b, '.-');
plot(tsa, x1a, 'o');
xlabel('Times [s]', 'fontsize',14);
grid on
title('State x1');

subplot(1, 2, 2);
hold on
plot(tsb, x2b, '.-');
plot(tsa, x2a, 'o');
legend(['grid_integrator', 'grid_control']);
xlabel('Times [s]', 'fontsize',14);
title('State x2');
grid on


  [tsol, usol] = sol.sample(u, 'grid','integrator','refine',100);

  figure();
  hold on
  plot(tsol,usol);
  title('Control signal');
  xlabel('Times [s]');
  grid on

  [tsc, x1c] = sol.sample(x1, 'grid','integrator', 'refine',100);

  figure();
  hold on
  plot(tsc, x1c, '-');
  plot(tsa, x1a, 'o');
  plot(tsb, x1b, '.');
  xlabel('Times [s]');
  grid on


