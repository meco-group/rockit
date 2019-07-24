from .stage_options import GridControl, GridIntegrator
import numpy as np
from casadi import vertcat


class OcpxSolution:
    def __init__(self,sol):
        self.sol = sol

    def sample(self, stage, expr, grid):
        if isinstance(grid,GridControl):
            return self._grid_control(stage, expr, grid)
            
        elif isinstance(grid,GridIntegrator):
            return self._grid_integrator(stage, expr, grid)
        else:
            raise Exception("Unknown grid option")

    def _grid_control(self,stage, expr, grid):
        sub_expr = []
        sub_expr.append(stage._constr_apply(expr,x=stage._method.X[0],u=stage._method.U[0]))
        for k in range(stage._method.N):
            sub_expr.append(stage._constr_apply(expr,x=stage._method.X[k],u=stage._method.U[k]))
        sub_expr.append(stage._constr_apply(expr,x=stage._method.X[-1],u=stage._method.U[-1]))
        res = [self.sol.value(elem) for elem in sub_expr]
        time = np.linspace(stage.t0,stage.tf,stage._method.N+1)
        return time, np.array(res)

    def _grid_integrator(self,stage, expr, grid):
        sub_expr = []
        sub_expr.append(stage._constr_apply(expr,x=stage._method.xk[0],u=stage._method.U[0]))
        for k in range(stage._method.N):
            for l in range(stage._method.M):
                sub_expr.append(stage._constr_apply(expr,x=stage._method.xk[k*stage._method.M + l],u=stage._method.U[k]))
        sub_expr.append(stage._constr_apply(expr,x=stage._method.xk[-1],u=stage._method.U[-1]))
        res = [self.sol.value(elem) for elem in sub_expr]
        time = np.linspace(stage.t0,stage.tf,stage._method.N*stage._method.M+1)
        return time, np.array(res)