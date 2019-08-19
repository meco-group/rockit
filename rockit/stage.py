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

from casadi import MX, substitute, Function, vcat, depends_on, vertcat, jacobian, veccat, jtimes
from .freetime import FreeTime
from .direct_method import DirectMethod
from .multiple_shooting import MultipleShooting
from collections import defaultdict
from .casadi_helpers import get_meta
from contextlib import contextmanager

class Stage:
    """
        A stage is defined on a time domain and has particular system dynamics
        associated with it.
    """
    def __init__(self, parent=None, t0=0, T=1):
        """Create an Optimal Control Problem stage.
        
        Only call this constructer when you need abstract stages,
        ie stages that are not associated with an :obj:`~rockit.ocp.Ocp`.
        For other uses, see :obj:`~rockit.stage.Stage.stage`.

        Parameters
        ----------
        parent : float or :obj:`~rockit.stage.Stage`, optional
            Parent Stage to which 
            Default: None
        t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
            Starting time of the stage
            Default: 0
        T : float or :obj:`~rockit.freetime.FreeTime`, optional
            Total horizon of the stage
            Default: 1

        Examples
        --------

        >>> stage = Stage()
        """
        self.states = []
        self.qstates = []
        self.controls = []
        self.algebraics = []
        self.parameters = defaultdict(list)
        self.variables = defaultdict(list)

        self.master = parent.master if parent else None
        self.parent = parent

        self._param_vals = dict()
        self._state_der = dict()
        self._alg = []
        self._constraints = defaultdict(list)
        self._objective = 0
        self._initial = dict()
        self._T = T
        self._t0 = t0
        self.t0_init = t0.T_init if isinstance(t0, FreeTime) else t0
        self.T_init = T.T_init if isinstance(T, FreeTime) else T
        self._placeholders = dict()
        self._placeholder_callbacks = dict()
        self._create_variables(t0, T)
        self._stages = []
        self._method = DirectMethod()

    def stage(self, template=None, **kwargs):
        """Create a new :obj:`~rockit.stage.Stage` and add it as to the :obj:`~rockit.ocp.Ocp`.

        Parameters
        ----------
        template : :obj:`~rockit.stage.Stage`, optional
            A stage to copy from. Will not be modified.
        t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
            Starting time of the stage
            Default: 0
        T : float or :obj:`~rockit.freetime.FreeTime`, optional
            Total horizon of the stage
            Default: 1

        Returns
        -------
        s : :obj:`~rockit.stage.Stage`
            New stage
        """
        if template:
            s = template.clone(self, **kwargs)
        else:
            s = Stage(self, **kwargs)
        self._stages.append(s)
        self._set_transcribed(False)
        return s

    def state(self, n_rows=1, n_cols=1, quad=False):
        """Create a state.
        You must supply a derivative for the state with :obj:`~rockit.stage.Stage.set_der`

        Parameters
        ----------
        n_rows : int, optional
            Number of rows
            Default: 1
        n_cols : int, optional
            Number of columns
            Default: 1

        Returns
        -------
        s : :obj:`~casadi.MX`
            A CasADi symbol representing a state

        Examples
        --------

        Defining the first-order ODE :  :math:`\dot{x} = -x`
        
        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> ocp.set_der(x, -x)
        >>> ocp.set_initial(x, sin(ocp.t)) # Optional: give initial guess
        """
        # Create a placeholder symbol with a dummy name (see #25)
        x = MX.sym("x", n_rows, n_cols)
        if quad:
            self.qstates.append(x)
        else:
            self.states.append(x)
        self._set_transcribed(False)
        return x

    def algebraic(self, n_rows=1, n_cols=1):
        """Create an algebraic variable
        You must supply an algebraic relation with:obj:`~rockit.stage.Stage.set_alg`

        Parameters
        ----------
        n_rows : int, optional
            Number of rows
            Default: 1
        n_cols : int, optional
            Number of columns
            Default: 1

        Returns
        -------
        s : :obj:`~casadi.MX`
            A CasADi symbol representing an algebraic variable
        """
        # Create a placeholder symbol with a dummy name (see #25)
        z = MX.sym("z", n_rows, n_cols)
        self.algebraics.append(z)
        self._set_transcribed(False)
        return z

    def variable(self, n_rows=1, n_cols=1, grid = ''):
        # Create a placeholder symbol with a dummy name (see #25)
        v = MX.sym("v", n_rows, n_cols)
        self.variables[grid].append(v)
        self._set_transcribed(False)
        return v

    def parameter(self, n_rows=1, n_cols=1, grid = ''):
        """
        Create a parameter
        """
        # Create a placeholder symbol with a dummy name (see #25)
        p = MX.sym("p", n_rows, n_cols)
        self.parameters[grid].append(p)
        self._set_transcribed(False)
        return p

    def control(self, n_rows=1, n_cols=1, order=0):
        """Create a control signal to optimize for

        A control signal is parametrized as a piecewise polynomial.
        By default (order=0), it is piecewise constant.

        Parameters
        ----------
        n_rows : int, optional
            Number of rows
        n_cols : int, optional
            Number of columns
        order : int, optional
            Order of polynomial. order=0 denotes a constant.
        Returns
        -------
        s : :obj:`~casadi.MX`
            A CasADi symbol representing a control signal

        Examples
        --------

        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> u = ocp.control()
        >>> ocp.set_der(x, u)
        >>> ocp.set_initial(u, sin(ocp.t)) # Optional: give initial guess
        """

        if order >= 1:
            u = self.state(n_rows, n_cols)
            helper_u = self.control(n_rows=n_rows, n_cols=n_cols, order=order - 1)
            self.set_der(u, helper_u)
            return u

        u = MX.sym("u", n_rows, n_cols)
        self.controls.append(u)
        self._set_transcribed(False)
        return u

    def set_value(self, parameter, value):
        if self.master.is_transcribed:
            self._method.set_value(self, self.master.opti, parameter, value)            
        else:
            self._param_vals[parameter] = value

    def set_initial(self, var, value):
        self._initial[var] = value
        if self.master.is_transcribed:
            self._method.set_initial(self, self.master.opti, self._initial)

    def set_der(self, state, der):
        """Assign a right-hand side to a state derivative

        Parameters
        ----------
        state : `~casadi.MX`
            A CasADi symbol created with :obj:`~rockit.stage.Stage.state`.
        der : `~casadi.MX`
            A CasADi symbolic expression of the same size as `state`

        Examples
        --------

        Defining the first-order ODE :  :math:`\dot{x} = -x`
        
        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> ocp.set_der(x, -x)
        """
        self._set_transcribed(False)
        self._state_der[state] = der

    def add_alg(self, constr):
        self._set_transcribed(False)
        self._alg.append(constr)

    def der(self, expr):
        if depends_on(expr, self.u):
            raise Exception("Dependency on controls not supported yet for stage.der")
        ode = self._ode()
        return jtimes(expr, self.x, ode(x=self.x, u=self.u, z=self.z, p=self.p, t=self.t)["ode"])

    def integral(self, expr, grid='inf'):
        """Compute an integral or a sum

        Parameters
        ----------
        expr : :obj:`~casadi.MX`
            An expression to integrate over the state time domain (from t0 to tf=t0+T)
        grid : str
            Possible entries:
                inf: the integral is performed using the integrator defined for the stage
                control: the integral is evaluated as a sum on the control grid (start of each control interval)
                         Note that the final state is not included in this definition
        """
        if grid=='inf':
            I = self.state(quad=True)
            self.set_der(I, expr)
            return self.at_tf(I)
        else:
            return self._create_placeholder_expr(expr, 'integral_control')

    def subject_to(self, constr, grid=None):
        """Adds a constraint to the problem

        Parameters
        ----------
        constr : :obj:`~casadi.MX`
            A constrained expression. It should be a symbolic expression that depends
            on decision variables and features a comparison `==`, `<=`, `=>`.

            If `constr` is a signal (:obj:`~rockit.stage.Stage.is_signal`, depends on time)
            a path-constraint is assumed: it should hold over the entire stage horizon.

            If `constr` is not a signal (e.g. :obj:`~rockit.stage.Stage.at_t0`/:obj:`~rockit.stage.Stage.at_tf` was applied on states),
            a boundary constraint is assumed.
        grid : str
            A string containing the type of grid to constrain the problem
            Possible entries: 
                control: constraint at control interval edges
                inf: use mathematical guarantee for the whole control interval (only possible for polynomials of states and controls)
                integrator: constrain at integrator edges
                integrator_roots: constrain at integrator roots (e.g. collocation points excluding 0)

        Examples
        --------

        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> ocp.set_der(x, -x)
        >>> ocp.subject_to( x <= 3)             # path constraint
        >>> ocp.subject_to( ocp.at_t0(x) == 0)  # boundary constraint
        >>> ocp.subject_to( ocp.at_tf(x) == 0)  # boundary constraint
        """
        self._set_transcribed(False)
        if grid is None:
            grid = 'control' if self.is_signal(constr) else 'point'
        if grid not in ['point', 'control', 'inf', 'integrator', 'integrator_roots']:
            raise Exception("Invalid argument")
        if self.is_signal(constr):
            if grid == 'point':
                raise Exception("Got a signal expression for grid 'point'.")
        else:
            if grid != 'point': 
                raise Exception("Expected signal expression since grid '" + grid + "' was given.")
        
        args = {"grid": grid}
        self._constraints[grid].append((constr, get_meta(), args))

    def at_t0(self, expr):
        """Evaluate a signal at the start of the horizon

        Parameters
        ----------
        expr : :obj:`~casadi.MX`
            A symbolic expression that may depend on states and controls

        Returns
        -------
        s : :obj:`~casadi.MX`
            A CasADi symbol representing an evaluation at `t0`.

        Examples
        --------

        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> ocp.set_der(x, -x)
        >>> ocp.subject_to( ocp.at_t0(sin(x)) == 0)
        """
        return self._create_placeholder_expr(expr, 'at_t0')

    def at_tf(self, expr):
        """Evaluate a signal at the end of the horizon

        Parameters
        ----------
        expr : :obj:`~casadi.MX`
            A symbolic expression that may depend on states and controls

        Returns
        -------
        s : :obj:`~casadi.MX`
            A CasADi symbol representing an evaluation at `tf`.

        Examples
        --------

        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> ocp.set_der(x, -x)
        >>> ocp.subject_to( ocp.at_tf(sin(x)) == 0)
        """
        return self._create_placeholder_expr(expr, 'at_tf')

    def add_objective(self, term):
        """Add a term to the objective of the Optimal Control Problem

        Parameters
        ----------
        term : :obj:`~casadi.MX`
            A symbolic expression that may not depend directly on states and controls.
            Use :obj:`~rockit.stage.Stage.at_t0`/:obj:`~rockit.stage.Stage.at_tf`/:obj:`~rockit.stage.Stage.integral`
            to eliminate the time-dependence of states and controls.

        Examples
        --------

        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> ocp.set_der(x, -x)
        >>> ocp.add_objective( ocp.at_tf(x) )    # Mayer term
        >>> ocp.add_objective( ocp.integral(x) ) # Lagrange term

        """
        self._set_transcribed(False)
        self._objective = self._objective + term

    def method(self, method):
        """Specify the transcription method

        Note that, for multi-stage problems, each stages can have a different method specification.

        Parameters
        ----------
        method : :obj:`~casadi.MX`
            Instance of a subclass of :obj:`~rockit.direct_method.DirectMethod`.
            Will not be modified

        Examples
        --------

        >>> ocp = Ocp()
        >>> ocp.method(MultipleShooting())
        """
        from copy import deepcopy
        self._set_transcribed(False)
        self._method = deepcopy(method)
        self._method.register(self)

    def is_free_time(self):
        """Does the stage have a free horizon length T?

        Returns
        -------
        res : bool

        """
        return isinstance(self._T, FreeTime)

    def is_free_starttime(self):
        """Does the stage have a free horizon start time t0?

        Returns
        -------
        res : bool

        """
        return isinstance(self._t0, FreeTime)

    @property
    def x(self):
        return veccat(*self.states)

    @property
    def xq(self):
        return veccat(*self.qstates)

    @property
    def u(self):
        return veccat(*self.controls)

    @property
    def z(self):
        return veccat(*self.algebraics)

    @property
    def p(self):
        return veccat((*self.parameters['']+self.parameters['control']))

    @property
    def nx(self):
        return self.x.numel()

    @property
    def nz(self):
        return self.z.numel()

    @property
    def nu(self):
        return self.u.numel()

    @property
    def np(self):
        return self.p.numel()

    def is_signal(self, expr):
        """Does the expression represent a signal (does it depend on time)?

        Returns
        -------
        res : bool

        """
 
        return depends_on(expr, vertcat(self.x, self.u, self.t, vcat(self.variables['control']+self.variables['states'])))

    def _create_placeholder_expr(self, expr, callback_name):
        r = MX.sym("r_" + callback_name, MX(expr).sparsity())
        self._placeholders[r] = expr
        self._placeholder_callbacks[r] = callback_name
        return r

    def _bake_placeholders(self, method):
        placeholders = dict()
        ks = list(self._placeholders.keys())
        vs = [self._placeholders[k] for k in ks]

        #if depends_on(vcat(expr), vcat(subst_from)):

        #vs = substitute(vs, subst_from, subs_to)

        for k, v in zip(ks, vs):
            callback = getattr(method, 'fill_placeholders_' + self._placeholder_callbacks[k])
            placeholders[k] = callback(self, v)

        return placeholders

    def _create_variables(self, t0, T):
        self.T = self._create_placeholder_expr(0, 'T')
        self.t0 = self._create_placeholder_expr(0, 't0')
        self.tf = self.t0 + self.T

        self.t = MX.sym('t')

    # Internal methods
    def _ode(self):
        ode = veccat(*[self._state_der[k] for k in self.states])
        quad = veccat(*[self._state_der[k] for k in self.qstates])
        alg = veccat(*self._alg)
        return Function('ode', [self.x, self.u, self.z, self.p, self.t], [ode, alg, quad], ["x", "u", "z", "p", "t"], ["ode","alg","quad"])

    def _expr_apply(self, expr, **kwargs):
        """
        Substitute placeholder symbols with actual decision variables,
        or expressions involving decision variables
        """
        subst_from, subst_to = self._get_subst_set(**kwargs)
        return substitute([MX(expr)], subst_from, subst_to)[0]

    def _get_subst_set(self, **kwargs):
        subst_from = []
        subst_to = []
        if "t" in kwargs:
            subst_from.append(self.t)
            subst_to.append(kwargs["t"])
        if "x" in kwargs:
            subst_from.append(self.x)
            subst_to.append(kwargs["x"])
        if "z" in kwargs:
            subst_from.append(self.z)
            subst_to.append(kwargs["z"])
        if "xq" in kwargs:
            subst_from.append(self.xq)
            subst_to.append(kwargs["xq"])
        if "u" in kwargs:
            subst_from.append(self.u)
            subst_to.append(kwargs["u"])
        if "p" in kwargs and self.parameters['']:
            p = veccat(*self.parameters[''])
            subst_from.append(p)
            subst_to.append(kwargs["p"])
        if "p_control" in kwargs and self.parameters['control']:
            p = veccat(*self.parameters['control'])
            subst_from.append(p)
            subst_to.append(kwargs["p_control"])
        if "v" in kwargs and self.variables['']:
            v = veccat(*self.variables[''])
            subst_from.append(v)
            subst_to.append(kwargs["v"])
        if "v_control" in kwargs and self.variables['control']:
            v = veccat(*self.variables['control'])
            subst_from.append(v)
            subst_to.append(kwargs["v_control"])
        return (subst_from, subst_to)

    _constr_apply = _expr_apply

    def _set_transcribed(self, val):
        if self.master:
            self.master._is_transcribed = val

    @property
    def is_transcribed(self):
        if self.master:
            return self.master._is_transcribed
        else:
            return False

    def _transcribe(self):
        opti = self.master.opti
        placeholders = {}
        if self._method is not None:
            placeholders.update(self._method.transcribe(self, opti))

        for s in self._stages:
            stage_placeholders = s._transcribe()
            placeholders.update(stage_placeholders)
        return placeholders

    def clone(self, parent, **kwargs):
        ret = Stage(parent, **kwargs)
        from copy import copy, deepcopy

        # Placeholders need to be updated
        subst_from = list(self._placeholders.keys())
        subst_to = []
        for k in self._placeholders.keys():
            if k is self.T:  # T and t0 already have new placeholder symbols
                subst_to.append(ret.T)
            elif k is self.t0:
                subst_to.append(ret.t0)
            else:
                subst_to.append(MX.sym(k.name(), k.sparsity()))
        for k_old, k_new in zip(subst_from, subst_to):
            ret._placeholder_callbacks[k_new] = self._placeholder_callbacks[k_old]
            ret._placeholders[k_new] = self._placeholders[k_old]

        ret.states = copy(self.states)
        ret.controls = copy(self.controls)
        ret.parameters = deepcopy(self.parameters)
        ret.variables = deepcopy(self.variables)

        ret._param_vals = copy(self._param_vals)
        ret._state_der = copy(self._state_der)
        constr_types = self._constraints.keys()
        orig = [self._objective]
        for k in constr_types:
            orig.extend([c for c, _, _ in self._constraints[k]])
        res = substitute(orig, subst_from, subst_to)
        ret._objective = res[0]
        res = res[1:]
        ret._constraints = defaultdict(list)
        for k in constr_types:
            v = self._constraints[k]
            ret._constraints[k] = list(zip(res, [m for _, m, _ in v], [d for _, _, d in v]))
            res = res[len(v):]

        ret._initial = copy(self._initial)

        if 'T' not in kwargs:
            ret._T = copy(self._T)
            ret.T = substitute([MX(self.T)], subst_from, subst_to)[0]
        if 't0' not in kwargs:
            ret._t0 = copy(self._t0)
            ret.t0 = substitute([MX(self.t0)], subst_from, subst_to)[0]
        ret.tf = substitute([MX(self.tf)], subst_from, subst_to)[0]
        ret.t = self.t
        ret._method = deepcopy(self._method)

        ret._is_transcribed = False
        return ret

    def iter_stages(self, include_self=False):
        if include_self:
            yield self
        for s in self._stages:
            yield from s.iter_stages(include_self=True)
