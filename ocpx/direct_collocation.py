from .sampling_method import SamplingMethod
from casadi import sumsqr, horzcat, vertcat, linspace, substitute, MX, evalf,\
                   vcat, collocation_points, collocation_interpolators, hcat,\
                   repmat, DM
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
        self.degree = degree
        self.tau = collocation_points(degree, scheme)
        [self.C, self.D, self.B] = collocation_coeff(self.tau)
        self.A = []  # List that will hold algebraic decision variables 
        self.Z = []  # List that will hold helper collocation states

    def add_variables(self, stage, opti):
        self.add_time_variables(stage, opti)
        # We are creating variables in a special order such that the resulting constraint Jacobian
        # is block-sparse
        x = opti.variable(stage.nx)
        self.X.append(x)

        for k in range(self.N):
            self.U.append(opti.variable(stage.nu))
            self.Z.append([horzcat(x, opti.variable(stage.nx, self.degree))]+[opti.variable(stage.nx, self.degree+1) for i in range(self.M-1)])
            self.A.append([opti.variable(stage.nz, self.degree) for i in range(self.M)])
            x = opti.variable(stage.nx)
            self.X.append(x)


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
        self.xk = []
        self.q = 0

        for k in range(self.N):
            dt = (self.control_grid[k + 1] - self.control_grid[k])/self.M
            S = 1/repmat(hcat([dt**i for i in range(self.degree + 1)]), self.degree + 1, 1)
            t0 = self.control_grid[k]
            for i in range(self.M):
                self.xk.append(self.Z[k][i][:,0])
                self.poly_coeff.append(self.Z[k][i] @ (poly*S))
                for j in range(self.degree):
                    Pidot_j = self.Z[k][i] @ self.C[:,j]/ dt
                    res = f(x=self.Z[k][i][:, j+1], u=self.U[k], z=self.A[k][i][:,j], p=self.P, t=t0+dt*self.tau[j])
                    # Collocation constraints
                    opti.subject_to(Pidot_j == res["ode"])
                    self.q = self.q + res["quad"]*dt*self.B[j]
                    if stage.nz:
                        opti.subject_to(0 == res["alg"])
                t0 += dt

                # Continuity constraints
                x_next = self.X[k + 1] if i==self.M-1 else self.Z[k][i+1][:,0]
                opti.subject_to(self.Z[k][i] @ self.D == x_next)

            for c, meta, _ in stage._path_constraints_expr():  # for each constraint expression
                # Add it to the optimizer, but first make x,u concrete.
                opti.subject_to(self.eval_at_control(stage, c, k), meta=meta)

        for c, meta, _ in stage._path_constraints_expr():  # for each constraint expression
            # Add it to the optimizer, but first make x,u concrete.
            opti.subject_to(self.eval_at_control(stage, c, -1), meta=meta)

        for c, meta, _ in stage._boundary_constraints_expr():  # Append boundary conditions to the end
            opti.subject_to(self.eval(stage, c), meta=meta)

    def set_initial(self, stage, opti):
        super().set_initial(stage, opti)
        for k in range(self.N):
            x0 = DM(opti.debug.value(self.X[k], opti.initial()))
            for e in self.Z[k]:
                opti.set_initial(e, repmat(x0, 1, e.shape[1]//x0.shape[1]))
