#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

from .stage import Stage
from .solution import OcpSolution
from .direct_method import OptiWrapper
from .placeholders import TranscribedPlaceholders

class Ocp(Stage):
    def __init__(self, **kwargs):
        """Create an Optimal Control Problem environment

        Parameters
        ----------
        t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
            Starting time of the optimal control horizon
            Default: 0
        T : float or :obj:`~rockit.freetime.FreeTime`, optional
            Total horizon of the optimal control horizon
            Default: 1

        Examples
        --------

        >>> ocp = Ocp()
        """
        Stage.__init__(self, **kwargs)
        self.master = self
        # Flag to make solve() faster when solving a second time
        # (e.g. with different parameter values)
        self._is_transcribed = False
        self._transcribed_placeholders = TranscribedPlaceholders()
        self.opti = OptiWrapper(self)

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

    def _transcribe(self):
        if not self.is_transcribed:
            self.opti.subject_to()
            self.opti.clear_objective()
            self._transcribe_recurse()
            self._set_transcribed(True)
            self._transcribed_placeholders.clear()
            self.opti.transcribe_placeholders(self.placeholders_transcribed)

    @property
    def placeholders_transcribed(self):
        self._transcribe()
        if self._transcribed_placeholders.is_dirty:
            self._placeholders_transcribe_recurse(self._transcribed_placeholders.pool)
            self._transcribed_placeholders.is_dirty = False
        return self._transcribed_placeholders

    @property
    def non_converged_solution(self):
        return OcpSolution(self.opti.non_converged_solution, self.master)

    def solve(self):
        self._transcribe()
        return OcpSolution(self.opti.solve(), self)

    def callback(self, fun):
        self.opti.callback(lambda iter : fun(iter, OcpSolution(self.opti.non_converged_solution, self.master)))

    @property
    def debug(self):
        self.opti.debug

    def solver(self, solver, solver_options={}):
        self.opti.solver(solver, solver_options)

    def show_infeasibilities(self, *args, **kwargs):
        self.opti.debug.show_infeasibilities(*args, **kwargs)
