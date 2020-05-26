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


% Garden swing
% ============
% How should you move your body to get a swing moving?
% Demo with Lagrangian mechanics


import rockit.*


%   *  
%   :\ L
%   : \   |
%    a \  |   H, M
%       \ |
%  Q _____. Centre of mass
%    | h
%    |
%    | l, m
%    | F


ocp = Ocp('T',12.0);

L = 2; % [m]
H = 1; % [m]
h = 0.5; % [m]
l = 0.5; % [m]
M = 60; % [kg]
m = 10; % [kg]
g = 9.81;

Rn = m*g*h;

alpha  = ocp.state();  % Rope angle wrt vertical (drawn pos)
beta   = ocp.state();  % Human angle wrt rope (drawn pos)
gamma  = ocp.state();  % Lower leg wrt to upper leg (draw +pi/2)
dalpha = ocp.state();  % Rope angle wrt vertical (drawn pos)
dbeta  = ocp.state();  % Human angle wrt rope (drawn pos)
dgamma = ocp.state();  % Lower leg wrt to upper leg (draw +pi/2)

R = ocp.control(); % Restore torque
T = ocp.control(); % Knee torque

delta = alpha-beta;
ddelta = dalpha-dbeta;

tau = delta-gamma+pi/2;
dtau = ddelta-dgamma;

q = [alpha; beta; gamma];
dq = [dalpha; dbeta; dgamma];

C = [L*sin(alpha); -L*cos(alpha)];
Q = C+[-h*cos(delta); -h*sin(delta)];
F = Q+[l*sin(tau);-l*cos(tau)];
c = (Q+F)/2;

dC = jtimes(C,q,dq);
dQ = jtimes(Q,q,dq);
dc = jtimes(c,q,dq);

% Modeling with Lagrange mechanics
E_kin = 0.5*dtau^2*(m*l^2/12) + 0.5*ddelta^2*(M*H^2/12) + 0.5*M*sumsqr(dC) + 0.5*m*sumsqr(dc);
E_pot = M*g*C(2)+m*g*c(2);

Lag = E_kin - E_pot;

E_tot = E_kin + E_pot;

Lag_q = gradient(Lag,q);
Lag_dq = gradient(Lag,dq);

rhs = solve(jacobian(Lag_dq,dq),[0;R;T]+Lag_q-jtimes(Lag_dq,q,dq), 'symbolicqr');

ocp.set_der(alpha,  dalpha);
ocp.set_der(beta,   dbeta);
ocp.set_der(gamma,  dgamma);
ocp.set_der(dalpha, rhs(1));
ocp.set_der(dbeta,  rhs(2));
ocp.set_der(dgamma, rhs(3));

ocp.set_initial(gamma, pi/2);
ocp.set_initial(R, Rn);

% Initial constraints
ocp.subject_to(ocp.at_t0(alpha)==0);
ocp.subject_to(ocp.at_t0(beta)==0);
ocp.subject_to(ocp.at_t0(gamma)==pi/2);
ocp.subject_to(ocp.at_t0(dalpha)==0);
ocp.subject_to(ocp.at_t0(dbeta)==0);
ocp.subject_to(ocp.at_t0(dgamma)==0);

% Don't cheat with kinetic energy
ocp.subject_to(ocp.at_tf(dbeta)==0);
ocp.subject_to(ocp.at_tf(dgamma)==0);

ocp.subject_to(-Rn <= T<= Rn);
ocp.subject_to(-5*Rn <= R<= 5*Rn);
ocp.subject_to(-pi/2 <= alpha<=pi/2,'grid','inf');
ocp.subject_to(-pi/8 <= beta<=pi/8,'grid','inf');
ocp.subject_to(pi/4 <= gamma<=pi,'grid','inf');


% Minimal time

% More ideally, || E_tot ||_inf
ocp.add_objective(-ocp.at_tf(E_tot));

% Pick a solution method
opts = struct;
opts.expand = true;
opts.ipopt.mumps_scaling = 0;
opts.ipopt.nlp_scaling_method = 'none';
ocp.solver('ipopt', opts);

N = 40;

% Make it concrete for this ocp
ocp.method(DirectCollocation('N',N,'M',8));

% solve
sol = ocp.solve();

figure();

[ts,xs] = sol.sample(alpha,'grid','integrator','refine',10);
plot(ts,xs);
hold on
[ts,xs] = sol.sample(beta,'grid','integrator','refine',10);
plot(ts,xs);
[ts,xs] = sol.sample(gamma,'grid','integrator','refine',10);
plot(ts,xs);
legend('alpha','beta','gamma');

figure();
[ts,xs] = sol.sample(R,'grid','integrator','refine',10);
plot(ts,xs);
hold on
[ts,xs] = sol.sample(T,'grid','integrator','refine',10);
plot(ts,xs);
legend('R','T');

figure();
[ts,xs] = sol.sample(T,'grid','integrator','refine',10);
plot(ts,xs);


[~, Cs] = sol.sample(C,'grid','integrator');
[~, Qs] = sol.sample(Q,'grid','integrator');
[ts, Fs] = sol.sample(F,'grid','integrator');
figure();
for k = 1:length(ts)
    plot([0,Cs(k,1)],[0,Cs(k,2)],'k-o');
    hold on
    plot([Cs(k,1),Qs(k,1)],[Cs(k,2),Qs(k,2)],'k-o');
    plot([Fs(k,1),Qs(k,1)],[Fs(k,2),Qs(k,2)],'k-o');
    xlim([-2*L , 2*L]);
    ylim([-2*L, 2*L]);
    pause(0.03);
    clf();
end
axis square
