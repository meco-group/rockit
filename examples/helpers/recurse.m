function [] = recurse(template, x, u, parent_stage, branching_var, branchings, varargin)
  current_depth = 0;
  if length(varargin)==1
      current_depth = varargin{1};
  end
  if isempty(branchings)
      return;
  end
  non_anticipatory = {};
  for b=branchings{1}
    stage = parent_stage.stage(template, 't0',current_depth);
    stage.set_value(branching_var, b);
    recurse(template, x, u, stage, branching_var, branchings(2:end), current_depth+1);

    if current_depth==0
      stage.master.subject_to(stage.at_t0(x)==[1.0;0]);
    else
      stage.master.subject_to(stage.at_t0(x)==parent_stage.at_tf(x));
    end
    non_anticipatory{end+1} = stage.at_t0(u);
  end
  for i=1:length(non_anticipatory)-1
      stage.master.subject_to(non_anticipatory{i+1}==non_anticipatory{i});
  end
end