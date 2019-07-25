import numpy as np
from casadi import vertcat
from .stage_options import GridControl, GridIntegrator, GridIntegratorFine


class OcpxSolution:
    def __init__(self, nlpsol):
        """Wrap casadi.nlpsol to simplify access to numerical solution."""
        self.sol = nlpsol

    def sample(self, stage, expr, grid):
        """Sample expression at solution on a given grid.

        Returns a numpy.array with the numerical values of the
        expression at the grid points specifeid by the grid type.

        arguments:
        state -- an optimal control problem stage
        expr  -- arbitrary expression containing states, controls, ...
        grid  -- type of time grid to use for sampling,
        options are available in ocpx.stage_options
        """
        if isinstance(grid, GridControl):
            return self._grid_control(stage, expr, grid)

        elif isinstance(grid, GridIntegrator):
            return self._grid_integrator(stage, expr, grid)
        
        elif isinstance(grid, GridIntegratorFine):
            return self._grid_intg_fine(stage, expr, grid)
        else:
            raise Exception("Unknown grid option: {}".format(grid))

    def _grid_control(self, stage, expr, grid):
        """Evaluate expression at (N + 1) control points."""
        sub_expr = []
        for k in range(stage._method.N):
            sub_expr.append(stage._constr_apply(
                expr, x=stage._method.X[k], u=stage._method.U[k]))
        sub_expr.append(stage._constr_apply(
            expr, x=stage._method.X[-1], u=stage._method.U[-1]))
        res = [self.sol.value(elem) for elem in sub_expr]
        time = self.sol.value(stage._method.control_grid)
        return time, np.array(res)

    def _grid_integrator(self, stage, expr, grid):
        """Evaluate expression at (N*M + 1) integrator discretization points."""
        sub_expr = []
        for k in range(stage._method.N):
            for l in range(stage._method.M):
                sub_expr.append(stage._constr_apply(
                    expr, x=stage._method.xk[k * stage._method.M + l], u=stage._method.U[k]))
        sub_expr.append(stage._constr_apply(
            expr, x=stage._method.xk[-1], u=stage._method.U[-1]))
        res = [self.sol.value(elem) for elem in sub_expr]
        time = np.linspace(stage.t0, self.sol.value(stage._method.T),
                           stage._method.N * stage._method.M + 1)
        return time, np.array(res)

    def _grid_intg_fine(self, stage, expr, grid):
        """Evaluate expression at extra fine integrator discretization points."""
        N, M, T = stage._method.N, stage._method.M, stage._method.T
        sub_expr = []
        res = []
        fine = 10
        ts = np.linspace(0, self.sol.value(T)/N/M,fine+1)[0:-1]
        for k in range(N):
            for l in range(5*M):
                sub_expr.append(stage._constr_apply(
                    expr, x=stage._method.poly_coeff[k*5*M+l], u=stage._method.U[k]))
            cf = [self.sol.value(elem) for elem in sub_expr]

            for d in range(M):
                res.extend(cf[d*5]*ts**0 + cf[d*5+1]*ts**1 + cf[d*5+2]*ts**2 + cf[d*5+3]*ts**3 + cf[d*5+4]*ts**4)
            sub_expr = []

        ts = self.sol.value(T)/N/M
        d = M-1
        final_value = cf[d*5]*ts**0 + cf[d*5+1]*ts**1 + cf[d*5+2]*ts**2 + cf[d*5+3]*ts**3 + cf[d*5+4]*ts**4
 
        res.append(final_value)

        time = np.linspace(stage.t0, self.sol.value(T), fine * N * M + 1)
        return time, res
