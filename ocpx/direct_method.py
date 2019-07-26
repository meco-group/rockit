from casadi import Opti, jacobian, dot, hessian


class DirectMethod:
    """
    Base class for 'direct' solution methods for Optimal Control Problems:
      'first discretize, then optimize'
    """

    def __init__(self, solver, solver_options={}):
        self.opti = OptiWrapper()
        self.opti.solver(solver, solver_options)

    def spy_jacobian(self):
        import matplotlib.pylab as plt
        J = jacobian(self.opti.g, self.opti.x).sparsity()
        plt.spy(J)
        plt.title("Constraint Jacobian: " + J.dim(True))

    def spy_hessian(self):
        import matplotlib.pylab as plt
        lag = self.opti.f + dot(self.opti.lam_g, self.opti.g)
        H = hessian(lag, self.opti.x)[0].sparsity()
        plt.spy(H)
        plt.title("Lagrange Hessian: " + H.dim(True))



from casadi import substitute

class OptiWrapper(Opti):
    def set_ocp(self, ocp):
        self.ocp = ocp
        self.constraints = []
        self.objective = 0
        self.placeholders = None

    def subject_to(self, expr=None):
        if expr is None:
            self.constraints = []
        else:
            self.constraints.append(expr)

    def add_objective(self, expr):
        self.objective = self.objective + expr

    def clear_objective(self):
        self.objective = 0

    def solve(self, placeholders=None):
        if placeholders:
            print(placeholders)
            ks = list(placeholders.keys())
            vs = [placeholders[k] for k in ks]
            print(self.constraints[-3])
            res = substitute(self.constraints + [self.objective], ks, vs)
            print(res[-4])
            for c in res[:-1]:
                super().subject_to(c)
            super().minimize(res[-1])
            self.placeholders = placeholders
        return OptiSolWrapper(self, super().solve())

class OptiSolWrapper:
    def __init__(self, opti_wrapper, sol):
        self.opti_wrapper = opti_wrapper
        self.sol = sol

    def value(self, expr):
        placeholders = self.opti_wrapper.placeholders
        ks = list(placeholders.keys())
        vs = [placeholders[k] for k in ks]
        res = substitute([expr], ks, vs)[0]
        return self.sol.value(expr)