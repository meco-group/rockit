from .sampling_method import SamplingMethod
from casadi import sumsqr, horzcat, vertcat, linspace, substitute, MX, evalf,\
                   vcat, collocation_points, collocation_interpolators, hcat,\
                   repmat
try:
    from casadi import collocation_coeff
except:
    def collocation_coeff(tau):
        [C, D] = collocation_interpolators(tau)
        d = len(tau)
        tau = [0]+tau
        F = [None]*(d+1)
        # Construct polynomial basis
        for j in range(d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(d+1):
                if r != j:
                    p *= np.poly1d([1, -tau[r]]) / (tau[j]-tau[r])
            
            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            F[j] = pint(1.0)
    
        return (hcat(C[1:]), vcat(D), hcat(F[1:]))

import numpy as np

class DirectCollocation(SamplingMethod):
    def __init__(self, *args, degree=4, scheme='radau', **kwargs):
        SamplingMethod.__init__(self, *args, **kwargs)
        if not self.M == 1:
            raise NotImplementedError(
                "Direct Collocation not yet supported for M!=1")
        self.degree = degree
        self.tau = collocation_points(degree, scheme)
        [self.C, self.D, self.B] = collocation_coeff(self.tau)
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

        ps = []
        tau_root = [0] + self.tau
        # Construct polynomial basis
        for j in range(self.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(self.degree + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
            ps.append(hcat(p.coef[::-1]))
        poly = vcat(ps)

        
        self.poly_coeff = []
        self.xk = self.X
        self.q = 0

        for k in range(self.N):
            dt = self.control_grid[k + 1] - self.control_grid[k]
            S = 1/repmat(hcat([dt**i for i in range(self.degree + 1)]), self.degree + 1, 1)
            Z = horzcat(self.X[k], self.Z[k])
            self.poly_coeff.append(Z @ (poly*S))
            for j in range(self.degree):
                Pidot_j = Z @ self.C[:,j]/ dt
                res = f(x=self.Z[k][:, j], u=self.U[k], p=self.P, t=self.control_grid[k]+dt*self.tau[j])
                # Collocation constraints
                opti.subject_to(Pidot_j == res["ode"])
                self.q = self.q + res["quad"]*dt*self.B[j]

            # Continuity constraints
            opti.subject_to(Z @ self.D == self.X[k + 1])

            for c, meta, _ in stage._path_constraints_expr():  # for each constraint expression
                # Add it to the optimizer, but first make x,u concrete.
                opti.subject_to(self.eval_at_control(stage, c, k), meta=meta)

        for c, meta, _ in stage._path_constraints_expr():  # for each constraint expression
            # Add it to the optimizer, but first make x,u concrete.
            opti.subject_to(self.eval_at_control(stage, c, -1), meta=meta)

        for c, meta, _ in stage._boundary_constraints_expr():  # Append boundary conditions to the end
            opti.subject_to(self.eval(stage, c), meta=meta)