import numpy as np
from casadi import vertcat, vcat, DM, Function, hcat


class OcpxSolution:
    def __init__(self, nlpsol, stage):
        """Wrap casadi.nlpsol to simplify access to numerical solution."""
        self.sol = nlpsol
        self.stage = stage

    def __call__(self, stage):
        """Sample expression at solution on a given grid.

        Parameters
        ----------
        stage : :obj:`~ocpx.stage.Stage`
            An optimal control problem stage.
        """
        return OcpxSolution(self.sol, stage=stage)

    def sample(self, expr, grid, **kwargs):
        """Sample expression at solution on a given grid.

        Parameters
        ----------
        expr : :obj:`casadi.MX`
            Arbitrary expression containing states, controls, ...
        grid : `str`
            At which points in time to sample, options are
            'control' or 'integrator' (at integrator discretization
            level).
        refine : int, optional
            Refine grid by evaluation the polynomal of the integrater at
            intermediate points ("refine" points per interval).

        Returns
        -------
        time : numpy.ndarray
            Time from zero to final time, same length as res
        res : numpy.ndarray
            Numerical values of evaluated expression at points in time vector.

        Examples
        --------
        Assume an ocp with a stage is already defined.

        >>> sol = ocp.solve()
        >>> tx, xs = sol.sample(x, grid='control')
        """
        if grid == 'control':
            return self._grid_control(self.stage, expr, grid, **kwargs)
        elif grid == 'integrator':
            if 'refine' in kwargs:
                return self._grid_intg_fine(self.stage, expr, grid, **kwargs)
            else:
                return self._grid_integrator(self.stage, expr, grid, **kwargs)
        else:
            msg = "Unknown grid option: {}\n".format(grid)
            msg += "Options are: 'control' or 'integrator' with an optional extra refine=<int> argument."
            raise Exception(msg)

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
        time = self.sol.value(stage._method.control_grid)
        time = np.linspace(time[0], time[-1], stage._method.N * stage._method.M + 1)
        return time, np.array(res)

    def _grid_intg_fine(self, stage, expr, grid, refine):
        """Evaluate expression at extra fine integrator discretization points."""
        if stage._method.poly_coeff is None:
            msg = "No polynomal coefficients for the {} integration method".format(stage._method.intg)
            raise Exception(msg)
        N, M, T = stage._method.N, stage._method.M, stage._method.T

        expr_f = Function('expr', [stage.x, stage.u], [expr])

        sub_expr = []
        tlocal = np.linspace(0, self.sol.value(T) / N / M, refine + 1) 
        ts = DM(tlocal[:-1]).T
        for k in range(N):
            for l in range(M):
                coeff = stage._method.poly_coeff[k * M + l]
                tpower = vcat([ts**i for i in range(coeff.shape[1])])
                sub_expr.append(expr_f(stage._method.poly_coeff[k * M + l] @ tpower, stage._method.U[k]))

        ts = tlocal[-1]
        tpower = vcat([ts**i for i in range(coeff.shape[1])])
        sub_expr.append(expr_f(stage._method.poly_coeff[-1] @ tpower, stage._method.U[-1]))

        res = self.sol.value(hcat(sub_expr))

        time = self.sol.value(stage._method.control_grid)
        time = np.linspace(time[0], time[-1], refine * N * M + 1)
        return time, res
