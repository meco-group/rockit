from .stage_options import GridControl, GridIntegrator
import numpy as np

class OcpxSolution:
    def __init__(self,sol):
        self.sol = sol

    def sample(self, stage, expr, grid):
        if isinstance(grid,GridControl):
            return self._grid_control(stage, expr, grid)
            
        elif isinstance(grid,GridIntegrator):
            pass
        else:
            raise Exception("Unknown grid option")

    def _grid_control(self,stage, expr, grid):
        sub_expr = []
        for k in range(stage._method.N):
            sub_expr.append(stage._constr_apply(expr,x=stage._method.X[k],u=stage._method.U[k]))
        sub_expr.append(stage._constr_apply(expr,x=stage._method.X[-1],u=stage._method.U[-1]))
        res = [self.sol.value(elem) for elem in sub_expr]
        time = self.sol.value(stage._method.control_grid)
        return time, res

 #   def _grid_integrator(self,stage, expr, grid):
#
#3       sub_expr = []
#      for k in range(stage._method.N):
#            sub_expr.append(stage._constr_apply(expr,x=stage._method.X[k],u=stage._method.U[k]))
#        res = [self.sol.value(elem) for elem in sub_expr]
#        time = np.linspace(stage.t0,stage.tf,stage._method.N)
#        return time, res
