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

from casadi import vertcat
from .stage import Stage, transcribed
from .placeholders import TranscribedPlaceholders

class Ocp(Stage):
    def __init__(self,  t0=0, T=1, **kwargs):
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
        Stage.__init__(self,  t0=t0, T=T, **kwargs)
        self._master = self
        # Flag to make solve() faster when solving a second time
        # (e.g. with different parameter values)
        self._var_is_transcribed = False
        self._transcribed_placeholders = TranscribedPlaceholders()

    @transcribed
    def jacobian(self, with_label=False):
        return self._method.jacobian(with_label=with_label)

    @transcribed
    def hessian(self, with_label=False):
        return self._method.hessian(with_label=with_label)

    @transcribed
    def spy_jacobian(self):
        self._method.spy_jacobian()

    @transcribed
    def spy_hessian(self):
        self._method.spy_hessian()

    @transcribed
    def spy(self):
        import matplotlib.pylab as plt
        plt.subplots(1, 2, figsize=(10, 4))
        plt.subplot(1, 2, 1)
        self.spy_jacobian()
        plt.subplot(1, 2, 2)
        self.spy_hessian()

    @property
    def _transcribed(self):
        if self._is_original:
            if self._is_transcribed:
                return self._augmented
            else:

                import copy
                augmented = copy.deepcopy(self)
                augmented._master = augmented
                augmented._transcribed_placeholders = self._transcribed_placeholders
                augmented._var_original = self
                self._var_augmented = augmented
                augmented._placeholders = self._placeholders
                
                return self._augmented._transcribed
        else:
            self._transcribe()
            return self

    def _transcribe(self):
        if not self.is_transcribed:
            self._transcribed_placeholders.clear()
            self._placeholders_transcribe_recurse(1,self._transcribed_placeholders)
            self._transcribe_recurse()
            self._original._set_transcribed(True)

            self._transcribe_recurse(phase=2,placeholders=self.placeholders_transcribed)

    @property
    @transcribed
    def placeholders_transcribed(self):
        """
        May also be called after solving (issue #91)
        """
        if self._transcribed_placeholders.is_dirty:
            self._placeholders_transcribe_recurse(2, self._transcribed_placeholders)
            self._transcribed_placeholders.is_dirty = False
        return self._transcribed_placeholders

    @property
    def non_converged_solution(self):
        return self._method.non_converged_solution(self)

    @transcribed
    def solve(self):
        return self._method.solve(self)
 
    @transcribed
    def solve_limited(self):
        return self._method.solve_limited(self)

    def callback(self, fun):
        return self._method.callback(self, fun)

    @property
    def debug(self):
        self._method.debug

    def solver(self, solver, solver_options={}):
        self._method.solver(solver, solver_options)

    def show_infeasibilities(self, *args):
        self._method.show_infeasibilities(*args)

    def debugme(self,e):
        print(e,hash(e),e.__hash__())
        return e

    @property
    def gist(self):
        """Obtain an expression packing all information needed to obtain value/sample

        The composition of this array may vary between rockit versions

        Returns
        -------
        :obj:`~casadi.MX` column vector

        """
        return self._method.gist

    def to_function(self, name, args, results, *margs):
        return self._method.to_function(self, name, args, results, *margs)
