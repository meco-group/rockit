from .stage import Stage
from .ocpx_solution import OcpxSolution
from .direct_method import OptiWrapper

class OcpMultiStage(Stage):
    def __init__(self, **kwargs):
        Stage.__init__(self, self, **kwargs)
        # Flag to make solve() faster when solving a second time
        # (e.g. with different parameter values)
        self.is_transcribed = False
        self.opti = OptiWrapper()

    def spy_jacobian(self):
        self._method.spy_jacobian(self.opti)

    def spy_hessian(self):
        self._method.spy_hessian(self.opti)

    def spy(self):
        import matplotlib.pylab as plt
        plt.subplots(1, 2, figsize=(10, 4))
        plt.subplot(1, 2, 1)
        self.spy_jacobian()
        plt.subplot(1, 2, 2)
        self.spy_hessian()

    def solve(self):
        if not self.is_transcribed:
            self.opti.subject_to()
            self.opti.clear_objective()
            placeholders = self._transcribe()
            self.is_transcribed = True
            return OcpxSolution(self.opti.solve(placeholders=placeholders))
        else:
            return OcpxSolution(self.opti.solve())

    def solver(self, solver, solver_options={}):
        self.opti.solver(solver, solver_options)
