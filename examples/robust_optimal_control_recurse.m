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


% Robust Optimal Control (using a recursive formulation)
% ======================================================
% Use a scenario tree of disturbances delta
% for which all possible realizations still meet a constraint

% robust optimal control
import rockit.*

ocp = Ocp();

template = Stage('T',1);
x1 = template.state();
x2 = template.state();

x = [x1; x2];
u = template.state();

delta = template.parameter();

template.set_der(x1, x2);
template.set_der(x2, -0.1*(1-x1^2 + delta)*x2 - x1 + u + delta);

template.set_der(u, 0);

template.subject_to(-40 <= u <= 40);
template.subject_to(x1 >= -0.25);
L = template.variable();
template.add_objective(L);
template.subject_to(L>= sumsqr(x1-3));
bound = @(t) 2 + 0.1*cos(10*t);
template.subject_to(x1 <= bound(template.t));
template.method(MultipleShooting('N',20, 'M',1, 'intg','rk'));

branchings = {[-1,1],[-1,1],[-1,1]};

addpath([cd filesep 'helpers'])
recurse(template, x, u, ocp, delta, branchings);

ocp.solver('ipopt',struct('expand',true));

sol = ocp.solve();


figure();
hold on
ss = ocp.iter_stages();
for i=1:length(ss)
  s=ss{i};
  [ts, xsol] = sol(s).sample(x1, 'grid','integrator');
  plot(ts, xsol, '-o');
  plot(ts, bound(ts),'r-');
end

grid on
xlabel('Time [s]')
ylabel('x1')
