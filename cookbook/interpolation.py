# Recipe: Smooth trajectory from set of points
from pylab import *
from casadi import interpolant

x = [0,1,2,3,4]
y = [4,6,1,1,2]
interp = interpolant('interp','bspline',[x],y,{"algorithm": "smooth_linear","smooth_linear_frac":0.1})
xs = np.linspace(0,4,1000)
plot(x,y,'o')
plot(xs,interp(xs))
show(block=True)
