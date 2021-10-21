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

% Motion planning with hyperplanes
% ================================

% Separating hyperplane is used to deal with
% convex polytopic obstacles


import rockit.*
ocp = Ocp('T',FreeTime(10.0));

% Bicycle model

x     = ocp.state();
y     = ocp.state();
theta = ocp.state();

delta = ocp.control();
V     = ocp.control();


ax = ocp.control('order',1);
ay = ocp.control('order',1);
b  = ocp.control('order',1);

L = 1;
r_veh = 0.5;

ocp.set_der(x, V*cos(theta));
ocp.set_der(y, V*sin(theta));
ocp.set_der(theta, V/L*tan(delta));

% Initial constraints
ocp.subject_to(ocp.at_t0(x)==0);
ocp.subject_to(ocp.at_t0(y)==0);
ocp.subject_to(ocp.at_t0(theta)==pi/2);

% Final constraint
ocp.subject_to(ocp.at_tf(x)==0);
ocp.subject_to(ocp.at_tf(y)==10);

ocp.set_initial(x,0);
ocp.set_initial(y,ocp.t);
ocp.set_initial(theta, pi/2);
ocp.set_initial(V,1);

ocp.subject_to(0 <= V<=1);
ocp.subject_to( -pi/6 <= delta<= pi/6);

points = [[-0.3;5],[1;5],[1;7],[-0.3;7]];

ocp.subject_to(ax*x+ay*y>=b + r_veh,'grid','control');

for p=points
  ocp.subject_to(ax*p(1)+ay*p(2)<=b);
end

ocp.subject_to(ax^2+ay^2<=1);


% Minimal time
ocp.add_objective(ocp.T);
ocp.add_objective(ocp.integral(x^2));

% Pick a solution method
ocp.solver('ipopt');

% Make it concrete for this ocp
ocp.method(MultipleShooting('N',20,'M',1,'intg','rk'));

% solve
sol = ocp.solve();


ocp.spy();

figure();
[~, xs] = sol.sample(x, 'grid','control');
[~, ys] = sol.sample(y, 'grid','control');
plot(xs, ys,'bo');
hold on

[~, xs] = sol.sample(x, 'grid','integrator','refine',10);
[~, ys] = sol.sample(y, 'grid','integrator','refine',10);
plot(xs, ys, '-');

[~, xs] = sol.sample(x, 'grid','integrator');
[~, ys] = sol.sample(y, 'grid','integrator');
plot(xs, ys, 'b.');

if size(points,2)>0
     plot([points(1,:) points(1,1)],[points(2,:) points(2,1)],'r-');
end

axis equal
xlim([-2.5, 2.5])
ylim([-1, 11])

[ts, axs] = sol.sample(ax, 'grid','integrator');
[ts, ays] = sol.sample(ay, 'grid','integrator');
[ts, bs] = sol.sample(b, 'grid','integrator');

xx = linspace(-2.5,2.5,101);
for i=1:length(ts)
    x = xs(i);
    y = ys(i);
    ax = axs(i);
    ay = ays(i);
    b = bs(i);
    
    plot(xx,-ax/ay*xx + b/ay);
    tfs = linspace(0,2*pi,100);
    plot(x+r_veh*cos(tfs), y+r_veh*sin(tfs),'r')
    pause(0.5)
    refresh
end