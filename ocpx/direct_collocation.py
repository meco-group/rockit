from .sampling_method import SamplingMethod
from casadi import sumsqr, horzcat, vertcat, linspace, substitute, MX, evalf,\
                   vcat, collocation_points, collocation_interpolators, hcat,\
                   repmat
import numpy as np

class DirectCollocation(SamplingMethod):
    def __init__(self, *args, degree=4, scheme='radau', **kwargs):
        SamplingMethod.__init__(self, *args, **kwargs)
        if not self.M == 1:
            raise NotImplementedError(
                "Direct Collocation not yet supported for M!=1")
        self.degree = degree
        self.tau = collocation_points(degree, scheme)
        [self.C, self.D] = collocation_interpolators(self.tau)

        self.Z = []  # List that will hold helper collocation states

    def add_variables(self, stage, opti):
        self.add_time_variables(stage, opti)
        # We are creating variables in a special order such that the resulting constraint Jacobian
        # is block-sparse
        self.X.append(opti.variable(stage.nx))

        for k in range(self.N):
            self.U.append(opti.variable(stage.nu))
            self.X.append(opti.variable(stage.nx))
            self.Z.append(opti.variable(stage.nx, self.degree))

    def add_constraints(self, stage, opti):
        # Obtain the discretised system
        f = stage._ode()

        if stage.is_free_time():
            opti.subject_to(self.T >= 0)

        self.poly_coeff = []
        self.xk = self.X

        for k in range(self.N):
            dt = self.control_grid[k + 1] - self.control_grid[k]
            Z = horzcat(self.X[k], self.Z[k])
            for j in range(self.degree):
                Pidot_j = Z @ vcat(self.C[j + 1]) / dt
                # Collocation constraints
                opti.subject_to(Pidot_j == f(
                    x=self.Z[k][:, j], u=self.U[k], p=self.P)["ode"])

            # Continuity constraints
            opti.subject_to(Z @ vcat(self.D) == self.X[k + 1])

            for c in stage._path_constraints_expr():  # for each constraint expression
                # Add it to the optimizer, but first make x,u concrete.
                opti.subject_to(self.eval_at_control(stage, c, k))

        for c in stage._boundary_constraints_expr():  # Append boundary conditions to the end
            opti.subject_to(self.eval(stage, c))