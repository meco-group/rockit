%% Problem specification

import rockit.*

% Start an optimal control environment with a time horizon of 10 seconds
%  (free time problems can be configured with `FreeTime(initial_guess)`)
ocp = Ocp('T',10);

% Define two scalar states (vectors and matrices also supported)
x1 = ocp.state();
x2 = ocp.state();

% Define one piecewise constant control input
%  (use extra arguments `'order',1` for piecewise linear)
u = ocp.control();

% Specify differential equations for states
%  (time dependency supported with `ocp.t`,
%   DAEs also supported with `ocp.algebraic` and `add_alg`)
ocp.set_der(x1, (1 - x2^2) * x1 - x2 + u);
ocp.set_der(x2, x1);

% Lagrange objective term
ocp.add_objective(ocp.integral(x1^2 + x2^2 + u^2));
% Mayer objective term
ocp.add_objective(ocp.at_tf(x1^2));

% Path constraints
%  (must be valid on the whole time domain running from `t0` to `tf=t0+T`,
%   grid options available such as `'grid','inf'`)
ocp.subject_to(x1 >= -0.25);
ocp.subject_to(-1 <= u <= 1 );

% Boundary constraints
ocp.subject_to(ocp.at_t0(x1) == 0);
ocp.subject_to(ocp.at_t0(x2) == 1);

%% Solving the problem-

% Pick an NLP solver backend
%  (CasADi `nlpsol` plugin):
ocp.solver('ipopt');

% Pick a solution method
%  N -- number of control intervals
%  M -- number of integration steps per control interval
method = MultipleShooting('N',10,'M',2, 'intg','rk');
%method = DirectCollocation('N',10,'M',2);

ocp.method(method);

% Solve
sol = ocp.solve();

%% Post-processing

% Show structure
ocp.spy()

% Sample a state/control or expression thereof on a grid
[tsa, x1a] = sol.sample(x1, 'grid', 'control');
[tsa, x2a] = sol.sample(x2, 'control');

[tsb, x1b] = sol.sample(x1, 'integrator');
[tsb, x2b] = sol.sample(x2, 'integrator');


figure()
subplot(1, 2, 1)
hold on
plot(tsb, x1b, '.-')
plot(tsa, x1a, 'o')
xlabel('Times [s]')
grid on
title('State x1')

subplot(1, 2, 2)
hold on
plot(tsb, x2b, '.-')
plot(tsa, x2a, 'o')
legend('grid_{integrator}', 'grid_{control}')
xlabel('Times [s]')
title('State x2')
grid on


% Refine the grid for a more detailed plot
[tsol, usol] = sol.sample(u, 'integrator','refine',100);

figure()
plot(tsol,usol)
title('Control signal')
xlabel('Times [s]')
grid on

[tsc, x1c] = sol.sample(x1, 'integrator','refine',100);

figure()
hold on
plot(tsc, x1c, '-')
plot(tsa, x1a, 'o')
plot(tsb, x1b, '.')
xlabel('Times [s]')
grid on
