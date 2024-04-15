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
from casadi import vertcat, vcat, which_depends, MX, substitute, integrator, Function, depends_on
from .stage import Stage, transcribed
from .placeholders import TranscribedPlaceholders
from .casadi_helpers import vvcat, rockit_pickle_context, rockit_unpickle_context
from .external.manager import external_method
from .direct_method import DirectMethod
class Ocp(Stage):
    def __init__(self,  t0=0, T=1, scale=1, **kwargs):
        """Create an Optimal Control Problem environment

        Parameters
        ----------
        t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
            Starting time of the optimal control horizon
            Default: 0
        T : float or :obj:`~rockit.freetime.FreeTime`, optional
            Total horizon of the optimal control horizon
            Default: 1
        scale: float, optional
               Typical time scale

        Examples
        --------

        >>> ocp = Ocp()
        """
        Stage.__init__(self,  t0=t0, T=T, scale=scale, **kwargs)
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
        
    def transcribe(self,**kwargs):
        self._untranscribe()
        self._transcribe(**kwargs)

    def _transcribe(self,**kwargs):
        if not self.is_transcribed:
            self._transcribed_placeholders.clear()
            self._transcribe_recurse(phase=0,**kwargs)
            self._placeholders_transcribe_recurse(1,self._transcribed_placeholders)
            self._transcribe_recurse(phase=1,**kwargs)
            self._original._set_transcribed(True)

            self._transcribe_recurse(phase=2,placeholders=self.placeholders_transcribed,**kwargs)
    
    def _untranscribe(self,**kwargs):
        if self.is_transcribed:
            self._transcribed_placeholders.clear()
            self._untranscribe_recurse(phase=0)
            self._placeholders_untranscribe_recurse(1)
            self._untranscribe_recurse(phase=1)
            self._original._set_transcribed(False)

            self._untranscribe_recurse(phase=2)

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
        self._set_transcribed(False)
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
    @transcribed
    def gist(self):
        """Obtain an expression packing all information needed to obtain value/sample

        The composition of this array may vary between rockit versions

        Returns
        -------
        :obj:`~casadi.MX` column vector

        """
        return self._method.gist

    @transcribed
    def to_function(self, name, args, results, *margs):
        return self._method.to_function(self, name, args, results, *margs)

    def is_sys_time_varying(self):
        # For checks
        self._ode()
        ode = vvcat([self._state_der[k] for k in self.states])
        alg = vvcat(self._alg)
        rhs = vertcat(ode,alg)
        return depends_on(rhs, self.t)

    def is_parameter_appearing_in_sys(self):
        # For checks
        self._ode()
        ode = vvcat([self._state_der[k] for k in self.states])
        alg = vvcat(self._alg)
        rhs = vertcat(ode,alg)
        pall = self.parameters['']+self.parameters['control']
        dep = [depends_on(rhs,p) for p in pall]
        return dep

    def sys_dae(self):
        # For checks
        self._ode()
        ode = vvcat([self._state_der[k] for k in self.states])
        alg = vvcat(self._alg)

        dae = {}
        dae["x"] = self.x
        dae["z"] = self.z

        t0 = MX.sym("t0")
        dt = MX.sym("dt")
        tau = MX.sym("tau")
        dae["t"] = self.t

        pall = self.parameters['']+self.parameters['control']

        rhs = vertcat(ode,alg)
        dep = [depends_on(rhs,p) for p in pall]

        p = vvcat([p for p,dep in zip(pall,dep) if dep])

        dae["ode"] = ode
        dae["alg"] = alg
        dae["p"] = p
        dae["u"] = self.u
        return dae

    def view_api(self, name):
        method = self._method
        self.method(external_method(name,method=method))
        self._transcribed
        return self._method

    def sys_simulator(self, intg='rk', intg_options=None):
        if intg_options is None:
            intg_options = {}
        intg_options = dict(intg_options)
        # For checks
        self._ode()
        ode = vvcat([self._state_der[k] for k in self.states])
        alg = vvcat(self._alg)

        dae = {}
        dae["x"] = self.x
        dae["z"] = self.z

        t0 = MX.sym("t0")
        dt = MX.sym("dt")
        tau = MX.sym("tau")
        dae["t"] = tau

        pall = self.parameters['']+self.parameters['control']
        
        #dep = which_depends(vertcat(ode,alg),vvcat(pall),1,True)


        rhs = vertcat(ode,alg)
        dep = [depends_on(rhs,p) for p in pall]

        p = vvcat([p for p,dep in zip(pall,dep) if dep])

        [ode,alg] = substitute([ode,alg],[self.t],[t0+tau*dt])
        dae["ode"] = dt*ode
        dae["alg"] = alg
        dae["p"] = vertcat(self.u, t0, dt, p)

        try:
            intg = integrator('intg',intg,dae,0,1,intg_options)
        except:
            intg_options["t0"] = 0
            intg_options["tf"] = 1
            intg = integrator('intg',intg,dae,intg_options)

        z_initial_guess = MX.sym("z",self.z.sparsity()) if self.nz>0 else MX(0,1)
        
        intg_out = intg(x0=self.x, p=dae["p"], z0=z_initial_guess)
        simulator = Function('simulator',
            [self.x, self.u, p, t0, dt, z_initial_guess],
            [intg_out["xf"], intg_out["zf"]],
            ["x","u","p","t0","dt","z_initial_guess"],
            ["xf","zf"])
        return simulator

    def save(self,name):
        self._untranscribe()
        import pickle
        with rockit_pickle_context():
            pickle.dump(self,open(name,"wb"))

    @staticmethod
    def load(name):
        import pickle
        with rockit_unpickle_context():
            return pickle.load(open(name,"rb"))