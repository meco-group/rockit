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

from casadi import MX, substitute, Function, vcat, depends_on, vertcat, jacobian, veccat, jtimes, hcat,\
                   linspace, DM, constpow, mtimes, vvcat, low, floor, hcat, horzcat, DM, is_equal, \
                   Sparsity
from .freetime import FreeTime
from .direct_method import DirectMethod
from .multiple_shooting import MultipleShooting
from .single_shooting import SingleShooting
from collections import defaultdict
from .casadi_helpers import DM2numpy, get_meta, merge_meta, HashDict, HashDefaultDict, HashOrderedDict, HashList, for_all_primitives
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np
from numpy import nan

def transcribed(func):
    def function_wrapper(self, *args, **kwargs):
        return func(self._transcribed, *args, **kwargs)
    function_wrapper._decorator_original = func
    function_wrapper.__doc__ = func.__doc__
    return function_wrapper

class Stage:
    """
        A stage is defined on a time domain and has particular system dynamics
        associated with it.

        Each stage has a transcription method associated with it.
    """
    def __init__(self, parent=None, t0=0, T=1, clone=False):
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
        self.states = HashList()
        self.qstates = HashList()
        self.controls = HashList()
        self.algebraics = HashList()
        self.parameters = defaultdict(HashList)
        self.variables = defaultdict(HashList)

        self._master = parent.master if parent else None
        self.parent = parent

        self._meta = HashDict()
        self._var_original = None
        self._var_augmented = None

        self._param_vals = HashDict()
        self._state_der = HashDict()
        self._state_next = HashDict()
        self._alg = []
        self._constraints = defaultdict(list)
        self._objective = 0
        self._initial = HashOrderedDict()

        self._placeholders = HashDict()
        self._offsets = HashDict()
        self._inf_inert = HashOrderedDict()
        self._inf_der = HashOrderedDict()
        self._t = self._create_placeholder_expr(0, 't')
        self._stages = []
        self._method = DirectMethod()
        self._t0 = t0
        self._T = T
        self._public_T  = self._create_placeholder_expr(0, 'T')
        self._public_t0 = self._create_placeholder_expr(0, 't0')
        self._tf = self.T + self.t0
    
    @property
    def master(self):
        return self._master

    @property
    def t(self):
        return self._t

    @property
    def T(self):
        return self._public_T

    @property
    def t0(self):
        return self._public_t0

    @property
    def tf(self):
        return self._tf

    def set_t0(self, t0):
        self._t0 = t0

    def set_T(self, T):
        self._T = T

    def _param_value(self, p):
        if p not in self._param_vals:
            raise Exception("You forgot to declare a value (using ocp.set_value) of the following parameter: " + str(self._meta[p]))
        else:
            return self._param_vals[p]

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
            s = template.clone(self, clone=True, **kwargs)
        else:
            s = Stage(self, **kwargs)
        self._stages.append(s)
        self._set_transcribed(False)
        return s

    def state(self, n_rows=1, n_cols=1, quad=False,meta=None):
        """Create a state.
        You must supply a derivative for the state with :obj:`~rockit.stage.Stage.set_der`

        Parameterrs
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
        import numpy
        # Create a placeholder symbol with a dummy name (see #25)

        name = "q"+str(len(self.qstates)+1) if quad else "x"+str(len(self.states)+1)
        x = MX.sym(name, n_rows, n_cols)
        meta = merge_meta(meta, get_meta())
        return self.register_state(x, meta=meta, quad=quad)
        
    def register_state(self, x, quad=False, meta=None):
        if isinstance(x, list):
            for e in x:
                self.register_state(e, quad=quad, meta=meta)
            return
        self._meta[x] = merge_meta(meta, get_meta())
        if quad:
            self.qstates.append(x)
        else:
            self.states.append(x)
        self._set_transcribed(False)
        return x

    def algebraic(self, n_rows=1, n_cols=1,meta=None):
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
        meta = merge_meta(meta, get_meta())
        return self.register_algebraic(z, meta=meta)

    def register_algebraic(self, z, meta=None):
        if isinstance(z, list):
            for e in z:
                self.register_algebraic(e, meta=meta)
            return
        self._meta[z] = merge_meta(meta, get_meta())
        self.algebraics.append(z)
        self._set_transcribed(False)
        return z

    def variable(self, n_rows=1, n_cols=1, grid = '',meta=None):
        """Create a variable

        Variables are unknowns in the Optimal Control problem
        for which we seek optimal values.

        Parameters
        ----------
        n_rows : int, optional
            Number of rows
        n_cols : int, optional
            Number of columns
        grid : string, optional
            Default is '', resulting in a single variable available
            over the whole optimal control horizon.
            For MultipleShooting, 'control' can be used to
            declare a variable that is unique to every control interval.


        Returns
        -------
        s : :obj:`~casadi.MX`
            A CasADi symbol representing a variable

        Examples
        --------

        >>> ocp = Ocp()
        >>> v = ocp.variable()
        >>> x = ocp.state()
        >>> ocp.set_der(x, v)
        >>> ocp.set_initial(v, 3)
        """
        # Create a placeholder symbol with a dummy name (see #25)
        v = MX.sym("v"+str(np.random.random(1)), n_rows, n_cols)
        meta = merge_meta(meta, get_meta())
        return self.register_variable(v, grid=grid, meta=meta)

    def register_variable(self, v, grid = '',meta=None):
        if isinstance(v, list):
            for e in v:
                self.register_variable(e)
            return
        self._meta[v] = merge_meta(meta, get_meta())
        self.variables[grid].append(v)
        self._set_transcribed(False)
        return v

    def parameter(self, n_rows=1, n_cols=1, grid = '',meta=None):
        """Create a parameter

        Parameters are symbols of an Optimal COntrol problem
        that are externally imposed, but not hardcoded.

        The advantage of parameters over simple numbers/numerical matrices comes
        when you need to solve multiple different Optimal Control problems.
        Parameters avoid the need to initialize new problems form scratch all the time;
        the problem becomes parametric.


        Parameters
        ----------
        n_rows : int, optional
            Number of rows
        n_cols : int, optional
            Number of columns
        grid : string, optional
            Default is '', resulting in a single parameter available
            over the whole optimal control horizon. 
            For MultipleShooting, 'control' can be used to
            declare a parameter that is unique to every control interval.

        Returns
        -------
        s : :obj:`~casadi.MX`
            A CasADi symbol representing a parameter

        Examples
        --------

        >>> ocp = Ocp()
        >>> p = ocp.parameter()
        >>> x = ocp.state()
        >>> ocp.set_der(x, p)
        >>> ocp.set_value(p, 3)
        """
        # Create a placeholder symbol with a dummy name (see #25)
        p = MX.sym("p", n_rows, n_cols)
        meta = merge_meta(meta, get_meta())
        return self.register_parameter(p, grid=grid, meta=meta)

    def register_parameter(self, p, grid='', meta=None):
        if isinstance(p,list):
            for e in p:
                self.register_parameter(e)
            return
        self._meta[p] = merge_meta(meta, get_meta())
        self.parameters[grid].append(p)
        self._set_transcribed(False)
        return p

    def control(self, n_rows=1, n_cols=1, order=0,meta=None):
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
        meta = merge_meta(meta, get_meta())
        self.register_control(u, meta=meta)
        return u

    def register_control(self, u, meta=None):
        if isinstance(u,list):
            for e in u:
                self.register_control(e)
            return
        self._meta[u] = merge_meta(meta, get_meta())
        self.controls.append(u)
        self._set_transcribed(False)
        return u

    def set_value(self, parameter, value):
        """Set a value for a parameter

        All variables must be given a value before an optimal control problem can be solved.

        Parameters
        ----------
        parameter : :obj:`~casadi.MX`
            The parameter symbol to initialize
        value : number
            The value


        Examples
        --------

        >>> ocp = Ocp()
        >>> p = ocp.parameter()
        >>> x = ocp.state()
        >>> ocp.set_der(x, p)
        >>> ocp.set_value(p, 3)
        """
        if self.master is not None and self.master.is_transcribed:
            def action(parameter, value):
                self._method.set_value(self, self.master._method, parameter, value)      
        else:
            def action(parameter, value):
                if parameter not in self._meta:
                    raise Exception("You attempted to set the value of a non-parameter: " + str(parameter))
                if not np.any([parameter in p for p in self.parameters.values()]):
                    raise Exception("You attempted to set the value of a non-parameter. Did you mean ocp.set_initial()? Got " + str(parameter))
                self._param_vals[parameter] = value
        for_all_primitives(parameter, value, action, "First argument to set_value must be a parameter or a simple concatenation of parameters", rhs_type=DM)


    def set_initial(self, var, value, priority=True):
        """Provide an initial guess

        Many Optimal Control solution methods are based on
        iterative numerical recipes.
        The initial guess, or starting point, may influence the
        convergence behavior and the quality of the solution.

        By default, all states, controls, and variables are initialized with zero.
        Use set_initial to provide a non-zero initial guess.

        Parameters
        ----------
        var : :obj:`~casadi.MX`
            The variable, state or control symbol (shape n-by-1) to initialize
        value : :obj:`~casadi.MX`
            The value to initialize with. Possibilities:
              * scalar number (repeated to fit the shape of `var` if needed)
              * numeric matrix of shape n-by-N or n-by-(N+1) in the case of MultipleShooting
              * CasADi symbolic expression dependent on ocp.t 

        Examples
        --------

        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> u = ocp.control()
        >>> ocp.set_der(x, u)
        >>> ocp.set_initial(u, 1)
        >>> ocp.set_initial(u, linspace(0,1,10))
        >>> ocp.set_initial(u, sin(ocp.t))
        """
        assert "opti" not in str(var)
        def action(var, value):
            if var not in self._meta and var not in self._placeholders:
                raise Exception("You attempted to set the initial value of an unknown symbol: " + str(var))
            if np.any([var in p for p in self.parameters.values()]):
                raise Exception("You attempted to set the initial value of a parameter. Did you mean ocp.set_value()? Got " + str(var))
            if var.is_scalar():
                # Auto-transpose
                if isinstance(value, np.ndarray) and len(value.shape)==1:
                    value = value.reshape((1, value.shape[0]))
                if isinstance(value, DM) and value.shape[0]==1 and value.shape[1]>1:
                    value = value.T
            self._initial[var] = value
            if priority:
                self._initial.move_to_end(var, last=False)
        for_all_primitives(var, value, action, "First argument to set_initial must be a variable/signal or a simple concatenation of variables/signals")
        if self.master is not None and self.master.is_transcribed:
            self._method.set_initial(self, self.master._method, self._initial)

    def set_der(self, state, der):
        """Assign a right-hand side to a state derivative

        Parameters
        ----------
        state : `~casadi.MX`
            A CasADi symbol created with :obj:`~rockit.stage.Stage.state`.
            May not be an indexed or sliced state
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
        assert not self._state_next
        def action(state, der):
            if state not in self.states and state not in self.qstates:
                raise Exception("You used set_der on a non-state: " + str(state))
            self._state_der[state] = der
        for_all_primitives(state, der, action, "First argument to set_der must be a state or a simple concatenation of states")

    def set_next(self, state, next):
        """Assign an update rule for a discrete state

        Parameters
        ----------
        state : `~casadi.MX`
            A CasADi symbol created with :obj:`~rockit.stage.Stage.state`.
        next : `~casadi.MX`
            A CasADi symbolic expression of the same size as `state`

        Examples
        --------

        Defining the first-order difference equation :  :math:`x^{+} = -x`
        
        >>> ocp = Ocp()
        >>> x = ocp.state()
        >>> ocp.set_next(x, -x)
        """
        self._set_transcribed(False)
        self._state_next[state] = next
        def action(state, next):
            self._state_next[state] = next
        for_all_primitives(state, next, action, "First argument to set_next must be a state or a simple concatenation of states")
        assert not self._state_der

    def add_alg(self, constr):
        self._set_transcribed(False)
        self._alg.append(constr)

    def der(self, expr):
        if depends_on(expr, self.u):
            raise Exception("Dependency on controls not supported yet for stage.der")
        ode = self._ode()
        return jtimes(expr, self.x, ode(x=self.x, u=self.u, z=self.z, p=vertcat(self.p, self.v), t=self.t)["ode"])

    def integral(self, expr, grid='inf'):
        """Compute an integral or a sum

        Parameters
        ----------
        expr : :obj:`~casadi.MX`
            An expression to integrate over the state time domain (from t0 to tf=t0+T)
        grid : str
            Possible entries:
                inf: the integral is performed using the integrator defined for the stage
                control: the integral is evaluated as a sum on the control grid (start of each control interval),
                         with each term of the sum weighted with the time duration of the interval.
                         Note that the final state is not included in this definition
        """
        if grid=='inf':
            return self._create_placeholder_expr(expr, 'integral')
        else:
            return self._create_placeholder_expr(expr, 'integral_control')

    def sum(self, expr, grid='inf'):
        """Compute a sum

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
        return self._create_placeholder_expr(expr, 'sum_control')

    def offset(self, expr, offset):
        """Get the value of a signal at control interval current+offset

        Parameters
        ----------
        expr : :obj:`~casadi.MX`
            An expression
        offset : (positive or negative) integer
        """
        if int(offset)!=offset:
            raise Exception("Integer expected")
        offset = int(offset)
        ret = MX.sym("offset", expr.shape)
        self._offsets[ret] = (expr, offset)
        return ret

    def next(self, expr):
        """Get the value of a signal at the next control interval

        Parameters
        ----------
        expr : :obj:`~casadi.MX`
            An expression
        """
        return self.offset(expr, 1)

    def inf_inert(self, expr):
        """Specify that expression should be treated as constant for grid=inf constraints
        """
        ret = MX.sym("inert", MX(expr).sparsity())
        self._inf_inert[ret] = expr
        return ret

    def inf_der(self, expr):
        """Specify that expression should be treated as constant for grid=inf constraints
        """
        ret = MX.sym("der", MX(expr).sparsity())
        self._inf_der[ret] = expr
        return ret

    def prev(self, expr):
        """Get the value of a signal at the previous control interval

        Parameters
        ----------
        expr : :obj:`~casadi.MX`
            An expression
        """
        return self.offset(expr, -1)

    def clear_constraints(self):
        """
        Remove any previously declared constraints from the problem
        """
        self._set_transcribed(False)
        self._constraints = defaultdict(list)

    def subject_to(self, constr, grid=None,include_first=True,include_last=True,meta=None):
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
        include_first : bool
            Enforce constraint also at t0
        include_last : bool or "auto"
            Enforce constraint also at tf
            "auto" mode will only enforce the constraint if it is not dependent on a control signal,
            since typically control signals are not defined at tf.

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
        #import ipdb; ipdb.set_trace()
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
        
        args = {"grid": grid, "include_last": include_last, "include_first": include_first}
        self._constraints[grid].append((constr, get_meta(meta), args))

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
        assert not self.is_signal(term), "An objective cannot be a signal. You must use ocp.integral or ocp.at_t0/tf to remove the time-dependence"
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
        template = self._method
        self._method = deepcopy(method)
        self._method.inherit(template)

    @property
    def objective(self):
        return self._objective

    @property
    def x(self):
        return vvcat(self.states)

    @property
    def xq(self):
        return vvcat(self.qstates)

    @property
    def u(self):
        if len(self.controls)==0: return MX(0, 1)
        return vvcat(self.controls)

    @property
    def z(self):
        return vvcat(self.algebraics)

    @property
    def p(self):
        arg = self.parameters['']+self.parameters['control']
        return MX(0, 1) if len(arg)==0 else vvcat(arg)

    @property
    def v(self):
        return vvcat(self.variables['']+self.variables['control'])

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

    @property
    def gist(self):
        """Obtain an expression packing all information needed to obtain value/sample

        The composition of this array may vary between rockit versions

        Returns
        -------
        :obj:`~casadi.MX` column vector

        """
        return self.master.gist

    def is_signal(self, expr):
        """Does the expression represent a signal (does it depend on time)?

        Returns
        -------
        res : bool

        """
 
        return depends_on(expr, vertcat(self.x, self.u, self.z, self.t, vcat(self.parameters['control']), vcat(self.variables['control']+self.variables['states']), vvcat(self._inf_der.keys())))

    def _create_placeholder_expr(self, expr, callback_name):
        """
        Placeholders are transcribed in two phases
           Phase 1: before any decision variables are created
             e.g. augmenting the state-space (ocp.integral)
           Phase 2: substituting to concrete decision variables

        """
        r = MX.sym("r_" + callback_name, MX(expr).sparsity())
        self._placeholders[r] = (callback_name, expr)
        if self.master is not None:
            self.master._transcribed_placeholders.mark_dirty()
        return r



                
    def _transcribe_placeholders(self, phase, method, placeholders):

        def normalize(out):
            if out is None:
                return None
            if isinstance(out, dict):
                return out
            return {"normal": out} 

        def prefix(d, prefix=None):
            if prefix is None or d is None:
                return d
            ret = {}
            for k,v in d.items():
                ret[prefix+"."+k] = v
            return ret

        def do(tag=None):
            ret = prefix(normalize(callback(phase, self, expr)),tag)
            if ret is not None:
                placeholders[phase][symbol] = ret
                if phase==2 and isinstance(expr,MX) and expr.is_symbolic() and symbol in placeholders[phase-1]:
                    placeholders[phase][expr] = ret
        # Fixed-point iteration:
        # Phase 1 may introduce extra placeholders
        while True:
            len_before = len(self._placeholders)
            for symbol, (species,expr) in list(self._placeholders.items()):
                if symbol not in placeholders[phase]:
                    callback = getattr(method, 'fill_placeholders_' + species)
                    if phase==2 and symbol in placeholders[phase-1]:
                        for tag, expr in placeholders[phase-1][symbol].items():
                            do(tag)
                            continue
                    do()

            len_after = len(self._placeholders)
            if len_before==len_after: break

    # Internal methods
    def _ode(self):
        der = []
        for k in self.states:
            try:
                der.append(self._state_der[k])
            except:
                raise Exception("ocp.set_der missing for state defined at " + str(self._meta[k]))
        ode = veccat(*der)
        der = []
        for k in self.qstates:
            try:
                der.append(self._state_der[k])
            except:
                raise Exception("ocp.set_der missing for quadrature state defined at " + str(self._meta[k]))
        quad = veccat(*der)
        alg = veccat(*self._alg)
        t = self.t
        if not depends_on(vertcat(ode,alg,quad),t):
            t = MX.sym('t', Sparsity(1, 1))
        ret = Function('ode', [self.x, self.u, self.z, vertcat(self.p, self.v), t], [ode, alg, quad], ["x", "u", "z", "p", "t"], ["ode","alg","quad"])
        assert not ret.has_free()
        return ret

    # Internal methods
    def _diffeq(self):
        val = []
        for k in self.states:
            try:
                val.append(self._state_next[k])
            except:
                raise Exception("ocp.set_next missing for state defined at " + str(self._meta[k]))
        next = veccat(*val)
        val = []
        for k in self.qstates:
            try:
                val.append(self._state_next[k])
            except:
                raise Exception("ocp.set_next missing for quadrature state defined at " + str(self._meta[k]))
        quad = veccat(*val)
        dt = MX(1,1)
        return Function('ode', [self.x, self.u, vertcat(self.p, self.v), self.t, dt], [next, MX(), quad, MX(0, 1), MX()], ["x0", "u", "p", "t0", "DT"], ["xf","poly_coeff","qf","zf","poly_coeff_z"])

    def _expr_apply(self, expr, **kwargs):
        """
        Substitute placeholder symbols with actual decision variables,
        or expressions involving decision variables
        """
        subst_from, subst_to = self._get_subst_set(**kwargs)
        temp = [(f,t) for f,t in zip(subst_from, subst_to) if f is not None and not f.is_empty() and t is not None]
        subst_from = [e[0] for e in temp]
        subst_to = [e[1] for e in temp]
        return substitute([MX(expr)], subst_from, subst_to)[0]

    def _get_subst_set(self, **kwargs):
        subst_from = []
        subst_to = []
        if "sub" in kwargs:
            subst_from += kwargs["sub"][0]
            subst_to += kwargs["sub"][1]
        if "t" in kwargs:
            subst_from.append(self.t)
            subst_to.append(kwargs["t"])
        if "x" in kwargs:
            subst_from.append(self.x)
            subst_to.append(kwargs["x"])
        if "z" in kwargs:
            subst_from.append(self.z)
            subst_to.append(kwargs["z"])
        if "t0" in kwargs and kwargs["t0"] is not None:
            subst_from.append(self.t0)
            subst_to.append(kwargs["t0"])
        if "T" in kwargs and kwargs["T"] is not None:
            subst_from.append(self.T)
            subst_to.append(kwargs["T"])
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
        if "v_states" in kwargs and self.variables['states']:
            v = veccat(*self.variables['states'])
            subst_from.append(v)
            subst_to.append(kwargs["v_states"])
        return (subst_from, subst_to)

    _constr_apply = _expr_apply

    @property
    def _is_original(self):
        return not self._var_original

    @property
    def _original(self):
        return self._var_original if self._var_original else self

    @property
    def _augmented(self):
        return self._var_augmented if self._var_augmented else self

    @property
    def _transcribed(self):
        if not self.is_transcribed:
            self.master._transcribe()
        if self._is_original:
            return self._augmented 
        else:
            return self

    def _set_transcribed(self, val):
        if self.master:
            if self._is_original:
                self.master._var_is_transcribed = val

    """
        In fact, both the original and the augmented should separately kee a transcribed flag
    """
    @property
    def _is_transcribed(self):
        if self._is_original:
            return self.master._var_is_transcribed 
        else:
            return self._original._is_transcribed

    @property
    def is_transcribed(self):
        if self.master:
            return self.master._is_transcribed
        else:
            return False

    def _transcribe_recurse(self, phase=1, **kwargs):
        if self._method is not None:
            if self is self.master:
                self._method.main_transcribe(self, phase=phase, **kwargs)
            self._method.transcribe(self, phase=phase, **kwargs)
        else:
            pass

        for s in self._stages:
            s._transcribe_recurse(phase=phase, **kwargs)

    def _placeholders_transcribe_recurse(self, phase, placeholders):
        if self._method is not None:
            self._method.transcribe_placeholders(phase, self, placeholders)

        for s in self._stages:
            s._placeholders_transcribe_recurse(phase, placeholders)

    def clone(self, parent, **kwargs):
        assert self._is_original
        ret = Stage(parent, **kwargs)
        from copy import copy, deepcopy

        # Placeholders need to be updated
        subst_from = list(self._placeholders.keys())
        subst_to = []
        for k in self._placeholders.keys():
            if is_equal(k, self.T):  # T and t0 already have new placeholder symbols
                subst_to.append(ret.T)
            elif is_equal(k, self.t0):
                subst_to.append(ret.t0)
            elif is_equal(k, self.t):
                subst_to.append(ret.t)
            else:
                subst_to.append(MX.sym(k.name(), k.sparsity()))
        for k_old, k_new in zip(subst_from, subst_to):
            ret._placeholders[k_new] = self._placeholders[k_old]

        ret.states = copy(self.states)
        ret.controls = copy(self.controls)
        ret.algebraics = copy(self.algebraics)
        ret.parameters = deepcopy(self.parameters)
        ret.variables = deepcopy(self.variables)

        ret._offsets = deepcopy(self._offsets)
        ret._param_vals = copy(self._param_vals)
        ret._state_der = copy(self._state_der)
        ret._alg = copy(self._alg)
        ret._state_next = copy(self._state_next)
        constr_types = self._constraints.keys()
        orig = []
        for k in constr_types:
            orig.extend([c for c, _, _ in self._constraints[k]])
        n_constr = len(orig)
        orig.append(self._objective)
        orig.extend(self._initial.keys())
        res = substitute(orig, subst_from, subst_to)
        ret._objective = res[n_constr]
        r = res[:n_constr]
        ret._constraints = defaultdict(list)
        for k in constr_types:
            v = self._constraints[k]
            ret._constraints[k] = list(zip(r, [merge_meta(m, get_meta()) for _, m, _ in v], [d for _, _, d in v]))
            r = r[len(v):]

        ret._initial = HashOrderedDict(zip(res[n_constr+1:], self._initial.values()))

        if "T" not in kwargs:
            ret._T = copy(self._T)
        if "t0" not in kwargs:
            ret._t0 = copy(self._t0)
        ret._method = deepcopy(self._method)
        ret._method.T = None
        ret._method.t0 = None
        ret._var_original = self._var_original

        ret._meta = self._meta

        ret._var_is_transcribed = False
        return ret

    def __deepcopy__(self, memo):
        # Get default deepcopy behaviour
        import copy
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method

        # Custom amendments
        cp._var_original = self
        self._var_augmented = cp

        cp._method = self._method

        return cp

    def iter_stages(self, include_self=False):
        if include_self:
            yield self
        for s in self._stages:
            for e in s.iter_stages(include_self=True): yield e

    @staticmethod
    def _parse_grid(grid):
        include_last = True
        include_first = True
        if grid.startswith('-'):
            grid = grid[1:]
            include_last = False
        if grid.endswith('-'):
            grid = grid[:-1]
            include_last = False
        return grid, include_first, include_last

    @transcribed
    def sample(self, expr, grid='control', **kwargs):
        """Sample expression symbolically on a given grid.

        Parameters
        ----------
        expr : :obj:`casadi.MX`
            Arbitrary expression containing states, controls, ...
        grid : `str`
            At which points in time to sample, options are
            'control' or 'integrator' (at integrator discretization
            level) or 'integrator_roots'.
        refine : int, optional
            Refine grid by evaluation the polynomal of the integrater at
            intermediate points ("refine" points per interval).

        Returns
        -------
        time : :obj:`casadi.MX`
            Time from zero to final time, same length as res
        res : :obj:`casadi.MX`
            Symbolically evaluated expression at points in time vector.

        Examples
        --------
        Assume an ocp with a stage is already defined.

        >>> sol = ocp.solve()
        >>> tx, xs = sol.sample(x, grid='control')
        """
        placeholders = self.master.placeholders_transcribed
        grid, include_first, include_last = self._parse_grid(grid)
        kwargs["include_first"] = include_first
        kwargs["include_last"] = include_last
        if grid == 'control':
            time, res = self._grid_control(self, expr, grid, **kwargs)
        elif grid == 'control-':
            time, res = self._grid_control(self, expr, grid, include_last=False, **kwargs)
        elif grid == 'integrator':
            if 'refine' in kwargs:
                time, res = self._grid_intg_fine(self, expr, grid, **kwargs)
            else:
                time, res = self._grid_integrator(self, expr, grid, **kwargs)
        elif grid == 'integrator_roots':
            time, res = self._grid_integrator_roots(self, expr, grid, **kwargs)
        else:
            msg = "Unknown grid option: {}\n".format(grid)
            msg += "Options are: 'control' or 'integrator' with an optional extra refine=<int> argument."
            raise Exception(msg)

        return placeholders(time), placeholders(res)

    def _grid_control(self, stage, expr, grid, include_first=True, include_last=True, transpose=False):
        """Evaluate expression at (N + 1) control points."""
        sub_expr = []
        ks = list(range(1, stage._method.N))
        if include_first:
            ks = [0]+ks
        if include_last:
            ks = ks+[-1]
        for k in ks:
            try:
                r = stage._method.eval_at_control(stage, expr, k)
            except IndexError as e:
                r = DM.nan(MX(expr).shape)
            sub_expr.append(r)
        cat = vcat if transpose else hcat
        res = cat(sub_expr)
        time = stage._method.control_grid
        return time, res

    def _grid_integrator(self, stage, expr, grid, include_first=True, include_last=True):
        """Evaluate expression at (N*M + 1) integrator discretization points."""
        sub_expr = []
        time = []
        assert include_first
        for k in range(stage._method.N):
            for l in range(stage._method.M):
                sub_expr.append(stage._method.eval_at_integrator(stage, expr, k, l))
            time.append(stage._method.integrator_grid[k])
        if include_last:
            sub_expr.append(stage._method.eval_at_control(stage, expr, -1))
        return vcat(time), hcat(sub_expr)


    def _grid_integrator_roots(self, stage, expr, grid, include_first=True, include_last=True):
        """Evaluate expression at integrator roots."""
        sub_expr = []
        tr = []
        assert include_first
        assert include_last
        for k in range(stage._method.N):
            for l in range(stage._method.M):
                for j in range(stage._method.xr[k][l].shape[1]):
                    sub_expr.append(stage._method.eval_at_integrator_root(stage, expr, k, l, j))
                tr.extend(stage._method.tr[k][l])
        return hcat(tr).T, hcat(sub_expr)

    def _grid_intg_fine(self, stage, expr, grid, refine, include_first=True, include_last=True):
        """Evaluate expression at extra fine integrator discretization points."""
        assert include_first
        assert include_last
        if stage._method.poly_coeff is None:
            msg = "No polynomal coefficients for the {} integration method".format(stage._method.intg)
            raise Exception(msg)
        N, M = stage._method.N, stage._method.M

        expr_f = Function('expr', [stage.t, stage.x, stage.z, stage.u, vertcat(stage.p, stage.v), stage.t0, stage.T], [expr])
        assert not expr_f.has_free(), str(expr_f.free_mx())

        time = stage._method.control_grid
        total_time = []
        sub_expr = []
        for k in range(N):
            t0 = time[k]
            dt = (time[k+1]-time[k])/M
            tlocal = linspace(MX(0), dt, refine + 1)
            assert tlocal.is_column()
            ts = tlocal[:-1,:]
            for l in range(M):
                local_t = t0+tlocal[:-1]
                total_time.append(local_t)
                coeff = stage._method.poly_coeff[k * M + l]
                tpower = hcat([constpow(ts,i) for i in range(coeff.shape[1])]).T
                if stage._method.poly_coeff_z:
                    coeff_z = stage._method.poly_coeff_z[k * M + l]
                    tpower_z = hcat([constpow(ts,i) for i in range(coeff_z.shape[1])]).T
                    z = mtimes(coeff_z,tpower_z)
                else:
                    z = nan

                pv = stage._method.get_p_sys(stage,k)
                sub_expr.append(stage._method.eval_at_integrator(stage, expr_f(local_t.T, mtimes(coeff,tpower), z, stage._method.U[k], pv, stage._method.t0, stage._method.T), k, l))
                t0+=dt

        ts = tlocal[-1,:]
        total_time.append(time[k+1])
        tpower = hcat([constpow(ts,i) for i in range(coeff.shape[1])]).T
        if stage._method.poly_coeff_z:
            tpower_z = hcat([constpow(ts,i) for i in range(coeff_z.shape[1])]).T
            z = mtimes(coeff_z,tpower_z)
        else:
            z = nan

        pv = stage._method.get_p_sys(stage,-1)
        sub_expr.append(stage._method.eval_at_integrator(stage, expr_f(time[k+1], mtimes(stage._method.poly_coeff[-1],tpower), z, stage._method.U[-1], pv, stage._method.t0, stage._method.T), k, l))

        return vcat(total_time), hcat(sub_expr)

    @transcribed
    def value(self, expr):
        """Get the value of an (non-signal) expression.

        Parameters
        ----------
        expr : :obj:`casadi.MX`
            Arbitrary expression containing no signals (states, controls) ...
        """
        placeholders = self.master.placeholders_transcribed
        return placeholders(self._method.eval(self, expr))

    @transcribed
    def discrete_system(self):
        """Hack"""
        return self._method.discrete_system(self)

    @transcribed
    def sampler(self, *args):
        """Returns a function that samples given expressions


        This function has two modes of usage:
        1)  sampler(exprs)  -> Python function
        2)  sampler(name, exprs, options) -> CasADi function

        Parameters
        ----------
        exprs : :obj:`casadi.MX` or list of :obj:`casadi.MX`
            List of arbitrary expression containing states, controls, ...
        name : `str`
            Name for CasADi Function
        options : dict, optional
            Options for CasADi Function

        Returns
        -------
        (gist, t) -> output
        mode 1 : Python Function
            Symbolically evaluated expression at points in time vector.
        mode 2 : :obj:`casadi.Function`
            Time from zero to final time, same length as res
        """

        numpy = True
        name = 'sampler'
        options = {}
        exprs = []
        ret_list = True
        if isinstance(args[0],str):
            name = args[0]
            exprs = args[1]
            if len(args)>=3: options = args[2]
            numpy = False
        else:
            exprs = args[0]
        if not isinstance(exprs, list):
            ret_list = False
            exprs = [exprs]
        t = MX.sym('t')

        """Evaluate expression at extra fine integrator discretization points."""
        if self._method.poly_coeff is None:
            msg = "No polynomal coefficients for the {} integration method".format(self._method.intg)
            raise Exception(msg)
        N, M = self._method.N, self._method.M

        expr_f = Function('expr', [self.t, self.x, self.z, self.u], exprs)
        assert not expr_f.has_free()

        time = vcat(self._method.integrator_grid)
        k = low(self._method.control_grid, t)        
        i = low(time, t)
        ti = time[i]
        tlocal = t-ti

        for c in self._method.poly_coeff:
            assert c.shape == self._method.poly_coeff[0].shape

        coeffs = hcat(self._method.poly_coeff)
        s = self._method.poly_coeff[0].shape[1]
        coeff = coeffs[:,(i*s+DM(range(s)).T)]

        tpower = constpow(tlocal,range(s))
        if self._method.poly_coeff_z:
            for c in self._method.poly_coeff_z:
                assert c.shape == self._method.poly_coeff_z[0].shape
            coeffs_z = hcat(self._method.poly_coeff_z)
            s_z = self._method.poly_coeff_z[0].shape[1]
            coeff_z = coeffs_z[:,i*s_z+DM(range(s_z)).T]
            tpower_z = constpow(tlocal,range(s_z))
            z = mtimes(coeff_z,tpower_z)
        else:
            z = nan

        Us = hcat(self._method.U)
        f = Function(name,[self.gist, t],expr_f.call([t, mtimes(coeff,tpower), z, Us[:,k]]), options)
        assert not f.has_free()

        if numpy:
            def wrapper(gist, t):
                """
                Parameters
                ----------
                gist : float vector
                    The gist of the solution, provided from `sol.gist` or
                    the evaluation of `ocp.gist`
                t : float or float vector
                    time or time-points to sample at

                Returns
                -------
                :obj:`np.array`

                """
                tdim = None if isinstance(t, float) or isinstance(t, int) or len(t.shape)==0 else DM(t).numel()
                t = DM(t)
                if t.is_column(): t = t.T
                res = f.call([gist, t])
                if ret_list:
                    return [DM2numpy(r, expr_f.size_out(i), tdim) for i,r in enumerate(res)]
                else:
                    return DM2numpy(res[0], expr_f.size_out(0), tdim)
            return wrapper
        else:
            return f
