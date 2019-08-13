from .sampling_method import SamplingMethod
from casadi import sumsqr, vertcat, linspace, substitute, MX, evalf, vcat, horzsplit, veccat, DM, repmat
from .splines import BSplineBasis, BSpline
from .casadi_helpers import reinterpret_expr
import numpy as np

class MultipleShooting(SamplingMethod):
    def __init__(self, *args, **kwargs):
        SamplingMethod.__init__(self, *args, **kwargs)

    def add_variables(self, stage, opti):
        self.add_time_variables(stage, opti)
        # We are creating variables in a special order such that the resulting constraint Jacobian
        # is block-sparse
        self.X.append(opti.variable(stage.nx) if stage.nx else MX(0, 1))

        V = []
        for v in stage.variables['']:
            V.append(opti.variable(v.shape[0], v.shape[1]))
        self.V = veccat(*V)


        V = []
        self.V_control = [[] for v in stage.variables['control']]

        for k in range(self.N):
            self.U.append(opti.variable(stage.nu) if stage.nu else MX(0, 1))
            self.X.append(opti.variable(stage.nx) if stage.nx else MX(0, 1))

            for i, v in enumerate(stage.variables['control']):
                self.V_control[i].append(opti.variable(v.shape[0], v.shape[1]))


    def add_constraints(self, stage, opti):
        # Obtain the discretised system
        F = self.discrete_system(stage)

        if stage.is_free_time():
            opti.subject_to(self.T >= 0)

        self.xk = []
        self.q = 0
        # we only save polynomal coeffs for runge-kutta4
        if stage._method.intg == 'rk':
            self.poly_coeff = []
        else:
            self.poly_coeff = None

        for k in range(self.N):
            FF = F(x0=self.X[k], u=self.U[k], t0=self.control_grid[k],
                   T=self.control_grid[k + 1] - self.control_grid[k], p=vertcat(veccat(*self.P), self.get_p_control_at(stage, k)))
            # Dynamic constraints a.k.a. gap-closing constraints
            opti.subject_to(self.X[k + 1] == FF["xf"])

            # Save intermediate info
            poly_coeff_temp = FF["poly_coeff"]
            xk_temp = FF["Xi"]
            self.q = self.q + FF["qf"]
            # we cannot return a list from a casadi function
            self.xk.extend([xk_temp[:, i] for i in range(self.M)])
            if self.poly_coeff is not None:
                self.poly_coeff.extend(horzsplit(poly_coeff_temp, poly_coeff_temp.shape[1]//self.M))

            for c, meta, arg in stage._path_constraints_expr():  # for each constraint expression
                # Add it to the optimizer, but first make x,u concrete.
                if arg["grid"] == "control":
                    opti.subject_to(self.eval_at_control(stage, c, k), meta=meta)
                elif arg["grid"] == "integrator":
                    raise Exception("Not implemented") 
                elif arg["grid"] == "inf":
                    for l in range(self.M):
                        coeff = stage._method.poly_coeff[k * self.M + l]
                        degree = coeff.shape[1]-1
                        basis = BSplineBasis([0]*(degree+1)+[1]*(degree+1),degree)
                        tscale = self.T / self.N / self.M
                        tpower = vcat([tscale**i for i in range(degree+1)])
                        coeff = coeff * repmat(tpower.T,stage.nx,1)
                        # TODO: bernstein transformation as function of degree
                        Poly_to_Bernstein_matrix_4 = DM([[1,0,0,0,0],[1,1.0/4, 0, 0, 0],[1, 1.0/2, 1.0/6, 0, 0],[1, 3.0/4, 1.0/2, 1.0/4, 0],[1, 1, 1, 1, 1]])
                        state_coeff = Poly_to_Bernstein_matrix_4 @ coeff.T
                        
                        statesize = [0] + [elem.nnz() for elem in stage.states]
                        statessizecum = np.cumsum(statesize)

                        subst_from = stage.states
                        state_coeff_split = horzsplit(state_coeff,statessizecum)
                        subst_to = [BSpline(basis,coeff) for coeff in state_coeff_split]
                        c_spline = reinterpret_expr(c, subst_from, subst_to)
                        opti.subject_to(self.eval_at_control(stage, c_spline, k), meta=meta)

        for c, meta, _ in stage._path_constraints_expr():  # for each constraint expression
            # Add it to the optimizer, but first make x,u concrete.
            opti.subject_to(self.eval_at_control(stage, c, -1), meta=meta)

        self.xk.append(self.X[-1])

        for c, meta, _ in stage._boundary_constraints_expr():  # Append boundary conditions to the end
            opti.subject_to(self.eval(stage, c), meta=meta)