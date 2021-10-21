import rockit.*

x = [0,1,2,3,4];
y = [4,6,1,1,2];
interp = casadi.interpolant('interp','bspline',{x},y,struct('algorithm','smooth_linear', 'smooth_linear_frac',0.1));
xs = linspace(0,4,1000);
plot(x,y,'o')
hold on
plot(xs,full(interp(xs)))