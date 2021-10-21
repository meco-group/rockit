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

% Robust Optimal Control
% ======================
% 
% Use a scenario tree of disturbances delta,;
% for which all possible realizations still meet a constraint;


% robust optimal control
import rockit.*

delta = 1;
ocp = Ocp();

addpath([cd filesep 'helpers'])
[stage1, x1, ut1] = robust_control_stages(ocp,delta,0);
ocp.subject_to(stage1.at_t0(x1)==[1.0;0]);

[stage2, x2, ut2] = robust_control_stages(ocp,-delta,0);
ocp.subject_to(stage2.at_t0(x2)==[1.0;0]);
ocp.subject_to(ut1 == ut2);

[stage3, x3, ut3] = robust_control_stages(ocp,delta,1);
ocp.subject_to(stage3.at_t0(x3)==stage1.at_tf(x1));

[stage4, x4, ut4] = robust_control_stages(ocp,-delta,1);
ocp.subject_to(stage4.at_t0(x4)==stage1.at_tf(x1));
ocp.subject_to(ut3 == ut4);

[stage5, x5, ut5] = robust_control_stages(ocp,delta,1);
ocp.subject_to(stage5.at_t0(x5)==stage2.at_tf(x2));

[stage6, x6, ut6] = robust_control_stages(ocp,-delta,1);
ocp.subject_to(stage6.at_t0(x6)==stage2.at_tf(x2));
ocp.subject_to(ut5 == ut6);

ocp.solver('ipopt');

sol = ocp.solve();

[ts1, xsol1] = sol(stage1).sample(x1(1), 'grid','integrator');
[ts2, xsol2] = sol(stage2).sample(x2(1), 'grid','integrator');
[ts3, xsol3] = sol(stage3).sample(x3(1), 'grid','integrator');
[ts4, xsol4] = sol(stage4).sample(x4(1), 'grid','integrator');
[ts5, xsol5] = sol(stage5).sample(x5(1), 'grid','integrator');
[ts6, xsol6] = sol(stage6).sample(x6(1), 'grid','integrator');

figure
hold on
plot(ts1, xsol1, '-o');
plot(ts2, xsol2, '-o');
plot(ts3, xsol3, '-o');
plot(ts4, xsol4, '-o');
plot(ts5, xsol5, '-o');
plot(ts6, xsol6, '-o');
  
bound = @(t) 2 + 0.1*cos(10*t);

plot(ts1, bound(ts1));
plot(ts3, bound(ts3));

legend('OCP trajectory, N=1','OCP trajectory, N=2','OCP trajectory, N=3','OCP trajectory, N=4', 'OCP trajectory, N=5','OCP trajectory, N=6');
xlabel('Time [s]');
ylabel('x1');