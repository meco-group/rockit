function [stage,x,u]=robust_control_stages(ocp,delta,t0)
  stage = ocp.stage('t0',t0, 'T',1);
  x = stage.state(2);
  u = stage.state();
  der_state = [x(2);-0.1*(1-x(1)^2 + delta)*x(2) - x(1) + u + delta];
  stage.set_der(x,der_state);
  stage.set_der(u, 0);
  stage.subject_to(-40 <= u <= 40);
  stage.subject_to(x(1) >= -0.25);
  L = stage.variable();
  stage.add_objective(L);
  stage.subject_to(L>= sumsqr(x(1)-3));
  bound = @(t) 2 + 0.1*cos(10*t);
  stage.subject_to(x(1) <= bound(stage.t));
  stage.method(rockit.MultipleShooting('N',20, 'M',1, 'intg','rk'));
  u = stage.at_t0(u);
end
