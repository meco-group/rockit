function []=plotme(iter, sol)
  figure();
  states = {x1,x2,u};
  label = {'x1','x2','u'};
  colors = {'r','g','b'};
  for i=1:3
    state = states{i};
    label = labels{i};
    color = colors{i};
    [tsa, x1a] = sol.sample(state, 'control')
    [tsb, x1b] = sol.sample(state, 'integrator')
    [tsc, x1c] = sol.sample(state, 'integrator', 'refine',100)
    plot(tsc, x1c, color+'-')
    plot(tsa, x1a, color+'o')
    plot(tsb, x1b, color+'.')
  xlabel('Times [s]')
  ylim([-0.3,1.1])
  title(['Iteration ' num2str(iter)])
  legend()
  grid true
end
