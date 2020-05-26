function [stage,p,v] = create_bouncing_ball_stage(ocp)
    % Create a bouncing ball stage.
    %
    % This function creates a stage of a bouncing ball that bounces no higher
    % than 5 meters above the ground.
    %
    % Returns
    % -------
    % stage : :obj:`~rockit.stage.Stage`
    %    An ocp stage describing the bouncing ball
    % p : :obj:`~casadi.MX`
    %    position variable
    % v : :obj:`~casadi.MX`
    %    velocity variable
    stage = ocp.stage('t0',rockit.FreeTime(0),'T',rockit.FreeTime(1));

    p = stage.state();
    v = stage.state();

    stage.set_der(p, v);
    stage.set_der(v, -9.81);

    stage.subject_to(stage.at_t0(v) >= 0);
    stage.subject_to(p >= 0);
    stage.method(rockit.MultipleShooting('N',1, 'M',20, 'intg','rk'));

end
