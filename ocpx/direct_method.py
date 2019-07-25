from casadi import Opti, jacobian, dot, hessian


class DirectMethod:
    """
    Base class for 'direct' solution methods for Optimal Control Problems:
      'first discretize, then optimize'
    """

    def __init__(self, solver, solver_options={}):
        self.opti = Opti()
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