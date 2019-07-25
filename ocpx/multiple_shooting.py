from .sampling_method import SamplingMethod
from casadi import sumsqr, vertcat, linspace, substitute, MX, evalf, vcat


class MultipleShooting(SamplingMethod):
    def __init__(self, *args, **kwargs):
        SamplingMethod.__init__(self, *args, **kwargs)
        self.X = []  # List that will hold N+1 decision variables for state vector
        self.U = []  # List that will hold N decision variables for control vector
        self.T = None
        self.t0 = None
        self.P = []
        self.poly_coeff = []  # Optional list to save the coefficients for a polynomial
        self.xk = []  # List for intermediate integrator states

    def transcribe(self, stage, opti):
        """
        Transcription is the process of going from a continous-time OCP to an NLP
        """
        self.add_variables(stage, opti)
        self.add_parameter(stage, opti)
        # Now that decision variables exist, we can bake the at_t0(...)/at_tf(...) expressions
        stage._bake(x0=self.X[0], xf=self.X[-1],
                    u0=self.U[0], uf=self.U[-1])
        self.add_constraints(stage, opti)
        self.add_objective(stage, opti)
        self.set_initial(stage, opti)
        self.set_parameter(stage, opti)

    def add_variables(self, stage, opti):
        # We are creating variables in a special order such that the resulting constraint Jacobian
        # is block-sparse
        self.X.append(opti.variable(stage.nx))
        if stage.is_free_time():
            self.T = opti.variable()
            opti.set_initial(self.T, stage._T.T_init)
        else:
            self.T = stage.T

        if stage.is_free_starttime():
            self.t0 = opti.variable()
            opti.set_initial(self.t0, stage._t0.T_init)
        else:
            self.t0 = stage.t0

        for k in range(self.N):
            self.U.append(opti.variable(stage.nu))
            self.X.append(opti.variable(stage.nx))

    def add_constraints(self, stage, opti):
        # Obtain the discretised system
        F = self.discrete_system(stage)

        # Create time grid (might be symbolic)
        self.control_grid = stage._expr_apply(
            linspace(MX(stage.t0), stage.tf, self.N + 1), T=self.T, t0=self.t0)

        if stage.is_free_time():
            opti.subject_to(self.T >= 0)

        self.poly_coeff = []
        self.xk = []

        for k in range(self.N):
            FF = F(x0=self.X[k], u=self.U[k], t0=self.control_grid[k],
                   T=self.control_grid[k + 1] - self.control_grid[k], p=vcat(self.P))
            # Dynamic constraints a.k.a. gap-closing constraints
            opti.subject_to(self.X[k + 1] == FF["xf"])

            # Save intermediate info
            self.poly_coeff.append(FF["poly_coeff"])
            x_temp = FF["Xi"]
            # we cannot return a list from a casadi function
            self.xk.extend([x_temp[:, i] for i in range(self.M)])

            for c in stage._path_constraints_expr():  # for each constraint expression
                # Add it to the optimizer, but first make x,u concrete.
                opti.subject_to(stage._constr_apply(
                    c, x=self.X[k], u=self.U[k], T=self.T, p=self.P,
					t=self.control_grid[k]))
        
        self.xk.append(self.X[-1])

        for c in stage._boundary_constraints_expr():  # Append boundary conditions to the end
            opti.subject_to(stage._constr_apply(c, p=self.P))

    def add_objective(self, stage, opti):
        opti.minimize(opti.f + stage._expr_apply(stage._objective, T=self.T))

    def set_initial(self, stage, opti):
        for var, expr in stage._initial.items():
            for k in range(self.N):
                opti.set_initial(stage._expr_apply(var, x=self.X[k], u=self.U[k]), opti.debug.value(
                    stage._expr_apply(expr, t=self.control_grid[k]), opti.initial()))
            opti.set_initial(stage._expr_apply(var, x=self.X[-1], u=self.U[-1]), opti.debug.value(
                stage._expr_apply(expr, t=self.control_grid[-1]), opti.initial()))

    def add_parameter(self, stage, opti):
        for p in stage.parameters:
            self.P.append(opti.parameter(p.shape[0], p.shape[1]))

    def set_parameter(self, stage, opti):
        for i, p in enumerate(stage.parameters):
            opti.set_value(self.P[i], stage._param_vals[p])
