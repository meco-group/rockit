from .stage import Stage
from .freetime import FreeTime
from casadi import hcat, substitute
from .ocpx_solution import OcpxSolution
from copy import deepcopy


class OcpMultiStage:
    def __init__(self):
        self.stages = []
        # Flag to make solve() faster when solving a second time
        # (e.g. with different parameter values)
        self.is_transcribed = False
        self._constraints = []

    def stage(self, prev_stage=None, **kwargs):
        if prev_stage is None:
            s = Stage(self, **kwargs)
        else:
            s = deepcopy(prev_stage)

        self.stages.append(s)
        return s

    def method(self, method):
        self._method = method

    def spy_jacobian(self):
        self._method.spy_jacobian()

    def spy_hessian(self):
        self._method.spy_hessian()

    def spy(self):
        import matplotlib.pylab as plt
        plt.subplots(1, 2, figsize=(10, 4))
        plt.subplot(1, 2, 1)
        self.spy_jacobian()
        plt.subplot(1, 2, 2)
        self.spy_hessian()

    def solve(self):
        opti = self._method.opti
        if not self.is_transcribed:
            constraints = self._constraints
            for s in self.stages:
                s._method.transcribe(s, opti)
                constraints = [s.subst_expr(c) for c in constraints]

            for c in constraints:
                opti.subject_to(c)

            self.is_transcribed = True

        return OcpxSolution(opti.solve())

    def free(self, T_init):
        return FreeTime(T_init)

    def subject_to(self, constr):
        """Set the constraint."""
        self._constraints.append(constr)
