classdef Stage < handle
  % 
  %         A stage is defined on a time domain and has particular system dynamics
  %         associated with it.
  % 
  %         Each stage has a transcription method associated with it.
  %     
  properties
    parent
  end
  methods
    function obj = Stage(varargin)
      % Create an Optimal Control Problem stage.
      % Arguments: parent=None, t0=0, T=1, scale=1, clone=False
      %         
      %         Only call this constructer when you need abstract stages,
      %         ie stages that are not associated with an :obj:`~rockit.ocp.Ocp`.
      %         For other uses, see :obj:`~rockit.stage.Stage.stage`.
      % 
      %         Parameters
      %         ----------
      %         parent : float or :obj:`~rockit.stage.Stage`, optional
      %             Parent Stage to which 
      %             Default: None
      %         t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Starting time of the stage
      %             Default: 0
      %         T : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Total horizon of the stage
      %             Default: 1
      %         scale: float, optional
      %                Typical time scale
      % 
      %         Examples
      %         --------
      % 
      %         >>> stage = Stage()
      %         
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.stage.Stage')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'parent','t0','T','scale','clone'});
      if isempty(kwargs)
        obj.parent = py.rockit.Stage(args{:});
      else
        obj.parent = py.rockit.Stage(args{:},pyargs(kwargs{:}));
      end
    end
    function varargout = set_t0(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'t0'});
      if isempty(kwargs)
        res = obj.parent.set_t0(args{:});
      else
        res = obj.parent.set_t0(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_T(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'T'});
      if isempty(kwargs)
        res = obj.parent.set_T(args{:});
      else
        res = obj.parent.set_T(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = stage(obj,varargin)
      % Create a new :obj:`~rockit.stage.Stage` and add it as to the :obj:`~rockit.ocp.Ocp`.
      % Arguments: template=None, kwargs
      % 
      %         Parameters
      %         ----------
      %         template : :obj:`~rockit.stage.Stage`, optional
      %             A stage to copy from. Will not be modified.
      %         t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Starting time of the stage
      %             Default: 0
      %         T : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Total horizon of the stage
      %             Default: 1
      % 
      %         Returns
      %         -------
      %         s : :obj:`~rockit.stage.Stage`
      %             New stage
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'template','kwargs'});
      if isempty(kwargs)
        res = obj.parent.stage(args{:});
      else
        res = obj.parent.stage(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = state(obj,varargin)
      % Create a state.
      % Arguments: n_rows=1, n_cols=1, quad=False, scale=1, meta=None
      %         You must supply a derivative for the state with :obj:`~rockit.stage.Stage.set_der`
      % 
      %         Parameters
      %         ----------
      %         n_rows : int, optional
      %             Number of rows
      %             Default: 1
      %         n_cols : int, optional
      %             Number of columns
      %             Default: 1
      %         scale : float or :obj:`~casadi.DM`, optional
      %             Provide a nominal value of the state for numerical scaling
      %             In essence, this has the same effect as defining x = scale*ocp.state(),
      %             except that set_initial(x, ...) keeps working
      %             Default: 1
      % 
      %         Returns
      %         -------
      %         s : :obj:`~casadi.MX`
      %             A CasADi symbol representing a state
      % 
      %         Examples
      %         --------
      % 
      %         Defining the first-order ODE :  :math:`\dot{x} = -x`
      %         
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, -x)
      %         >>> ocp.set_initial(x, sin(ocp.t)) # Optional: give initial guess
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'n_rows','n_cols','quad','scale','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.state(args{:});
      else
        res = obj.parent.state(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = register_state(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'x','quad','scale','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.register_state(args{:});
      else
        res = obj.parent.register_state(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = algebraic(obj,varargin)
      % Create an algebraic variable
      % Arguments: n_rows=1, n_cols=1, scale=1, meta=None
      %         You must supply an algebraic relation with:obj:`~rockit.stage.Stage.set_alg`
      % 
      %         Parameters
      %         ----------
      %         n_rows : int, optional
      %             Number of rows
      %             Default: 1
      %         n_cols : int, optional
      %             Number of columns
      %             Default: 1
      % 
      %         Returns
      %         -------
      %         s : :obj:`~casadi.MX`
      %             A CasADi symbol representing an algebraic variable
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'n_rows','n_cols','scale','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.algebraic(args{:});
      else
        res = obj.parent.algebraic(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = register_algebraic(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'z','scale','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.register_algebraic(args{:});
      else
        res = obj.parent.register_algebraic(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = variable(obj,varargin)
      % Create a variable
      % Arguments: n_rows=1, n_cols=1, grid=, order=0, scale=1, include_last=False, meta=None
      % 
      %         Variables are unknowns in the Optimal Control problem
      %         for which we seek optimal values.
      % 
      %         Parameters
      %         ----------
      %         n_rows : int, optional
      %             Number of rows
      %         n_cols : int, optional
      %             Number of columns
      %         grid : string, optional
      %             Default is '', resulting in a single variable available
      %             over the whole optimal control horizon.
      %             For MultipleShooting, 'control' can be used to
      %             declare a variable that is unique to every control interval.
      %             'bspline' indicates a bspline parametrization
      %         order : int, optional
      %             Relevant with grid='bspline'
      %         include_last : bool, optional
      %             Determines if a unique entry is foreseen at the tf edge.
      % 
      % 
      %         Returns
      %         -------
      %         s : :obj:`~casadi.MX`
      %             A CasADi symbol representing a variable
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> v = ocp.variable()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, v)
      %         >>> ocp.set_initial(v, 3)
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'n_rows','n_cols','grid','order','scale','include_last','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.variable(args{:});
      else
        res = obj.parent.variable(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = register_variable(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'v','grid','order','scale','include_last','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.register_variable(args{:});
      else
        res = obj.parent.register_variable(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = signal_shape(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'s'});
      if isempty(kwargs)
        res = obj.parent.signal_shape(args{:});
      else
        res = obj.parent.signal_shape(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = parameter(obj,varargin)
      % Create a parameter
      % Arguments: n_rows=1, n_cols=1, grid=, order=0, scale=1, include_last=False, meta=None
      % 
      %         Parameters are symbols of an Optimal COntrol problem
      %         that are externally imposed, but not hardcoded.
      % 
      %         The advantage of parameters over simple numbers/numerical matrices comes
      %         when you need to solve multiple different Optimal Control problems.
      %         Parameters avoid the need to initialize new problems form scratch all the time;
      %         the problem becomes parametric.
      % 
      % 
      %         Parameters
      %         ----------
      %         n_rows : int, optional
      %             Number of rows
      %         n_cols : int, optional
      %             Number of columns
      %         grid : string, optional
      %             Default is '', resulting in a single parameter available
      %             over the whole optimal control horizon. 
      %             For MultipleShooting, 'control' can be used to
      %             declare a parameter that is unique to every control interval.
      %             include_last determines if a unique entry is foreseen at the tf edge.
      % 
      %         Returns
      %         -------
      %         s : :obj:`~casadi.MX`
      %             A CasADi symbol representing a parameter
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> p = ocp.parameter()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, p)
      %         >>> ocp.set_value(p, 3)
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'n_rows','n_cols','grid','order','scale','include_last','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.parameter(args{:});
      else
        res = obj.parent.parameter(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = register_parameter(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'p','grid','order','scale','include_last','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.register_parameter(args{:});
      else
        res = obj.parent.register_parameter(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = control(obj,varargin)
      % Create a control signal to optimize for
      % Arguments: n_rows=1, n_cols=1, order=0, scale=1, meta=None
      % 
      %         A control signal is parametrized as a piecewise polynomial.
      %         By default (order=0), it is piecewise constant.
      % 
      %         Parameters
      %         ----------
      %         n_rows : int, optional
      %             Number of rows
      %         n_cols : int, optional
      %             Number of columns
      %         order : int, optional
      %             Order of polynomial. order=0 denotes a constant.
      %         scale : float or :obj:`~casadi.DM`, optional
      %             Provide a nominal value of the state for numerical scaling
      %             In essence, this has the same effect as defining u = scale*ocp.control(),
      %             except that set_initial(u, ...) keeps working
      %             Default: 1
      %         Returns
      %         -------
      %         s : :obj:`~casadi.MX`
      %             A CasADi symbol representing a control signal
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> u = ocp.control()
      %         >>> ocp.set_der(x, u)
      %         >>> ocp.set_initial(u, sin(ocp.t)) # Optional: give initial guess
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'n_rows','n_cols','order','scale','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.control(args{:});
      else
        res = obj.parent.control(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = register_control(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'u','scale','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.register_control(args{:});
      else
        res = obj.parent.register_control(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_value(obj,varargin)
      % Set a value for a parameter
      % Arguments: parameter, value
      % 
      %         All variables must be given a value before an optimal control problem can be solved.
      % 
      %         Parameters
      %         ----------
      %         parameter : :obj:`~casadi.MX`
      %             The parameter symbol to initialize
      %         value : number
      %             The value
      % 
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> p = ocp.parameter()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, p)
      %         >>> ocp.set_value(p, 3)
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'parameter','value'});
      if isempty(kwargs)
        res = obj.parent.set_value(args{:});
      else
        res = obj.parent.set_value(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_initial(obj,varargin)
      % Provide an initial guess
      % Arguments: var, value, priority=True
      % 
      %         Many Optimal Control solution methods are based on
      %         iterative numerical recipes.
      %         The initial guess, or starting point, may influence the
      %         convergence behavior and the quality of the solution.
      % 
      %         By default, all states, controls, and variables are initialized with zero.
      %         Use set_initial to provide a non-zero initial guess.
      % 
      %         Parameters
      %         ----------
      %         var : :obj:`~casadi.MX`
      %             The variable, state or control symbol (shape n-by-1) to initialize
      %         value : :obj:`~casadi.MX`
      %             The value to initialize with. Possibilities:
      %               * scalar number (repeated to fit the shape of `var` if needed)
      %               * numeric matrix of shape n-by-N or n-by-(N+1) in the case of MultipleShooting
      %               * CasADi symbolic expression dependent on ocp.t 
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> u = ocp.control()
      %         >>> ocp.set_der(x, u)
      %         >>> ocp.set_initial(u, 1)
      %         >>> ocp.set_initial(u, linspace(0,1,10))
      %         >>> ocp.set_initial(u, sin(ocp.t))
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'var','value','priority'});
      if isempty(kwargs)
        res = obj.parent.set_initial(args{:});
      else
        res = obj.parent.set_initial(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_der(obj,varargin)
      % Assign a right-hand side to a state derivative
      % Arguments: state, der, scale=1
      % 
      %         Parameters
      %         ----------
      %         state : `~casadi.MX`
      %             A CasADi symbol created with :obj:`~rockit.stage.Stage.state`.
      %             May not be an indexed or sliced state
      %         der : `~casadi.MX`
      %             A CasADi symbolic expression of the same size as `state`
      %         scale : extra scaling after scaling of state has been applied
      % 
      %         Examples
      %         --------
      % 
      %         Defining the first-order ODE :  :math:`\dot{x} = -x`
      %         
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, -x)
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'state','der','scale'});
      if isempty(kwargs)
        res = obj.parent.set_der(args{:});
      else
        res = obj.parent.set_der(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_next(obj,varargin)
      % Assign an update rule for a discrete state
      % Arguments: state, next
      % 
      %         Parameters
      %         ----------
      %         state : `~casadi.MX`
      %             A CasADi symbol created with :obj:`~rockit.stage.Stage.state`.
      %         next : `~casadi.MX`
      %             A CasADi symbolic expression of the same size as `state`
      % 
      %         Examples
      %         --------
      % 
      %         Defining the first-order difference equation :  :math:`x^{+} = -x`
      %         
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> ocp.set_next(x, -x)
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'state','next'});
      if isempty(kwargs)
        res = obj.parent.set_next(args{:});
      else
        res = obj.parent.set_next(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = add_alg(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'constr','scale'});
      if isempty(kwargs)
        res = obj.parent.add_alg(args{:});
      else
        res = obj.parent.add_alg(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = der(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.der(args{:});
      else
        res = obj.parent.der(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = integral(obj,varargin)
      % Compute an integral or a sum
      % Arguments: expr, grid=inf, refine=1
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`~casadi.MX`
      %             An expression to integrate over the state time domain (from t0 to tf=t0+T)
      %         grid : str
      %             Possible entries:
      %                 inf: the integral is performed using the integrator defined for the stage
      %                 control: the integral is evaluated as a sum on the control grid (start of each control interval),
      %                          with each term of the sum weighted with the time duration of the interval.
      %                          Note that the final state is not included in this definition
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr','grid','refine'});
      if isempty(kwargs)
        res = obj.parent.integral(args{:});
      else
        res = obj.parent.integral(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = sum(obj,varargin)
      % Compute a sum
      % Arguments: expr, grid=control, include_last=False
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`~casadi.MX`
      %             An expression to integrate over the state time domain (from t0 to tf=t0+T)
      %         grid : str
      %             Possible entries:
      %                 control: the integral is evaluated as a sum on the control grid (start of each control interval)
      %                          Note that the final state is not included in this definition
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr','grid','include_last'});
      if isempty(kwargs)
        res = obj.parent.sum(args{:});
      else
        res = obj.parent.sum(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = offset(obj,varargin)
      % Get the value of a signal at control interval current+offset
      % Arguments: expr, offset
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`~casadi.MX`
      %             An expression
      %         offset : (positive or negative) integer
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'expr','offset'});
      if isempty(kwargs)
        res = obj.parent.offset(args{:});
      else
        res = obj.parent.offset(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = next(obj,varargin)
      % Get the value of a signal at the next control interval
      % Arguments: expr
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`~casadi.MX`
      %             An expression
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.next(args{:});
      else
        res = obj.parent.next(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = inf_inert(obj,varargin)
      % Specify that expression should be treated as constant for grid=inf constraints
      % Arguments: expr
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.inf_inert(args{:});
      else
        res = obj.parent.inf_inert(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = inf_der(obj,varargin)
      % Specify that expression should be treated as constant for grid=inf constraints
      % Arguments: expr
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.inf_der(args{:});
      else
        res = obj.parent.inf_der(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = prev(obj,varargin)
      % Get the value of a signal at the previous control interval
      % Arguments: expr
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`~casadi.MX`
      %             An expression
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.prev(args{:});
      else
        res = obj.parent.prev(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = clear_constraints(obj,varargin)
      % 
      % Arguments: 
      %         Remove any previously declared constraints from the problem
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.clear_constraints(args{:});
      else
        res = obj.parent.clear_constraints(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = subject_to(obj,varargin)
      % Adds a constraint to the problem
      % Arguments: constr, grid=None, include_first=True, include_last=True, scale=1, refine=1, group_refine=<rockit.grouping_techniques.GroupingTechnique object at 0x7ff1ac339990>, group_dim=<rockit.grouping_techniques.GroupingTechnique object at 0x7ff1ac3399d0>, group_control=<rockit.grouping_techniques.GroupingTechnique object at 0x7ff1ac339e90>, meta=None
      % 
      %         Parameters
      %         ----------
      %         constr : :obj:`~casadi.MX`
      %             A constrained expression. It should be a symbolic expression that depends
      %             on decision variables and features a comparison `==`, `<=`, `=>`.
      % 
      %             If `constr` is a signal (:obj:`~rockit.stage.Stage.is_signal`, depends on time)
      %             a path-constraint is assumed: it should hold over the entire stage horizon.
      % 
      %             If `constr` is not a signal (e.g. :obj:`~rockit.stage.Stage.at_t0`/:obj:`~rockit.stage.Stage.at_tf` was applied on states),
      %             a boundary constraint is assumed.
      %         grid : str
      %             A string containing the type of grid to constrain the problem
      %             Possible entries: 
      %                 control: constraint at control interval edges
      %                 inf: use mathematical guarantee for the whole control interval (only possible for polynomials of states and controls)
      %                 integrator: constrain at integrator edges
      %                 integrator_roots: constrain at integrator roots (e.g. collocation points excluding 0)
      %         include_first : bool
      %             Enforce constraint also at t0
      %         include_last : bool or "auto"
      %             Enforce constraint also at tf
      %             "auto" mode will only enforce the constraint if it is not dependent on a control signal,
      %             since typically control signals are not defined at tf.
      %         refine : int, optional
      %             Refine grid used in constraining by a certain factor with respect to the control grid
      %         group_refine : GroupTechnique, optional
      %             Group constraints together along the refine axis
      %         group_dim : GroupTechnique, optional
      %             Group vector-valued constraints along the vector dimension into a scalar constraint
      %         group_control : GroupTechnique, optional
      %             Group constraints together along the control grid
      % 
      %         scale : float or :obj:`~casadi.DM`, optional
      %             Provide a nominal value for this constraint
      %             In essence, this has the same effect as dividing all sides of the constraints by scale
      %             Default: 1
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, -x)
      %         >>> ocp.subject_to( x <= 3)             # path constraint
      %         >>> ocp.subject_to( ocp.at_t0(x) == 0)  # boundary constraint
      %         >>> ocp.subject_to( ocp.at_tf(x) == 0)  # boundary constraint
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'constr','grid','include_first','include_last','scale','refine','group_refine','group_dim','group_control','meta'});
      meta = py.None;
      try
        st = dbstack('-completenames',1);
        if length(st)>0
          meta = struct('stacktrace', {{st(1)}});
          meta = pythoncasadiinterface.matlab2python(meta);
        end
      catch
      end
      kwargs = {kwargs{:} 'meta' meta};
      if isempty(kwargs)
        res = obj.parent.subject_to(args{:});
      else
        res = obj.parent.subject_to(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = at_t0(obj,varargin)
      % Evaluate a signal at the start of the horizon
      % Arguments: expr
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`~casadi.MX`
      %             A symbolic expression that may depend on states and controls
      % 
      %         Returns
      %         -------
      %         s : :obj:`~casadi.MX`
      %             A CasADi symbol representing an evaluation at `t0`.
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, -x)
      %         >>> ocp.subject_to( ocp.at_t0(sin(x)) == 0)
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.at_t0(args{:});
      else
        res = obj.parent.at_t0(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = at_tf(obj,varargin)
      % Evaluate a signal at the end of the horizon
      % Arguments: expr
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`~casadi.MX`
      %             A symbolic expression that may depend on states and controls
      % 
      %         Returns
      %         -------
      %         s : :obj:`~casadi.MX`
      %             A CasADi symbol representing an evaluation at `tf`.
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, -x)
      %         >>> ocp.subject_to( ocp.at_tf(sin(x)) == 0)
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.at_tf(args{:});
      else
        res = obj.parent.at_tf(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = add_objective(obj,varargin)
      % Add a term to the objective of the Optimal Control Problem
      % Arguments: term
      % 
      %         Parameters
      %         ----------
      %         term : :obj:`~casadi.MX`
      %             A symbolic expression that may not depend directly on states and controls.
      %             Use :obj:`~rockit.stage.Stage.at_t0`/:obj:`~rockit.stage.Stage.at_tf`/:obj:`~rockit.stage.Stage.integral`
      %             to eliminate the time-dependence of states and controls.
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> x = ocp.state()
      %         >>> ocp.set_der(x, -x)
      %         >>> ocp.add_objective( ocp.at_tf(x) )    # Mayer term
      %         >>> ocp.add_objective( ocp.integral(x) ) # Lagrange term
      % 
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'term'});
      if isempty(kwargs)
        res = obj.parent.add_objective(args{:});
      else
        res = obj.parent.add_objective(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = method(obj,varargin)
      % Specify the transcription method
      % Arguments: method
      % 
      %         Note that, for multi-stage problems, each stages can have a different method specification.
      % 
      %         Parameters
      %         ----------
      %         method : :obj:`~casadi.MX`
      %             Instance of a subclass of :obj:`~rockit.direct_method.DirectMethod`.
      %             Will not be modified
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         >>> ocp.method(MultipleShooting())
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'method'});
      if isempty(kwargs)
        res = obj.parent.method(args{:});
      else
        res = obj.parent.method(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = is_signal(obj,varargin)
      % Does the expression represent a signal (does it depend on time)?
      % Arguments: expr
      % 
      %         Returns
      %         -------
      %         res : bool
      % 
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.is_signal(args{:});
      else
        res = obj.parent.is_signal(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = is_parametric(obj,varargin)
      % Does the expression depend only on parameters?
      % Arguments: expr
      % 
      %         Returns
      %         -------
      %         res : bool
      % 
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.is_parametric(args{:});
      else
        res = obj.parent.is_parametric(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = clone(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'parent','kwargs'});
      if isempty(kwargs)
        res = obj.parent.clone(args{:});
      else
        res = obj.parent.clone(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = iter_stages(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'include_self'});
      if isempty(kwargs)
        res = obj.parent.iter_stages(args{:});
      else
        res = obj.parent.iter_stages(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = sample(obj,varargin)
      % Sample expression symbolically on a given grid.
      % Arguments: expr, grid=control, kwargs
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`casadi.MX`
      %             Arbitrary expression containing states, controls, ...
      %         grid : `str`
      %             At which points in time to sample, options are
      %             'control' or 'integrator' (at integrator discretization
      %             level), 'integrator_roots', 'gist'.            
      %         refine : int, optional
      %             Refine grid by evaluation the polynomal of the integrater at
      %             intermediate points ("refine" points per interval).
      % 
      %         Returns
      %         -------
      %         time : :obj:`casadi.MX`
      %             Time from zero to final time, same length as res
      %         res : :obj:`casadi.MX`
      %             Symbolically evaluated expression at points in time vector.
      % 
      %         Examples
      %         --------
      %         Assume an ocp with a stage is already defined.
      % 
      %         >>> sol = ocp.solve()
      %         >>> tx, xs = sol.sample(x, grid='control')
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr','grid','kwargs'});
      if isempty(kwargs)
        res = obj.parent.sample(args{:});
      else
        res = obj.parent.sample(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = value(obj,varargin)
      % Get the value of an (non-signal) expression.
      % Arguments: expr
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`casadi.MX`
      %             Arbitrary expression containing no signals (states, controls) ...
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.value(args{:});
      else
        res = obj.parent.value(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = initial_value(obj,varargin)
      % Get the value of an expression at initial guess
      % Arguments: expr
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`casadi.MX`
      %             Arbitrary expression containing no signals (states, controls) ...
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'expr'});
      if isempty(kwargs)
        res = obj.parent.initial_value(args{:});
      else
        res = obj.parent.initial_value(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = discrete_system(obj,varargin)
      % Hack
      % Arguments: 
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.discrete_system(args{:});
      else
        res = obj.parent.discrete_system(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = sampler(obj,varargin)
      % Returns a function that samples given expressions
      % Arguments: args
      % 
      % 
      %         This function has two modes of usage:
      %         1)  sampler(exprs)  -> Python function
      %         2)  sampler(name, exprs, options) -> CasADi function
      % 
      %         Parameters
      %         ----------
      %         exprs : :obj:`casadi.MX` or list of :obj:`casadi.MX`
      %             List of arbitrary expression containing states, controls, ...
      %         name : `str`
      %             Name for CasADi Function
      %         options : dict, optional
      %             Options for CasADi Function
      % 
      %         Returns
      %         -------
      %         (gist, t) -> output
      %         mode 1 : Python Function
      %             Symbolically evaluated expression at points in time vector.
      %         mode 2 : :obj:`casadi.Function`
      %             Time from zero to final time, same length as res
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'args'});
      if isempty(kwargs)
        res = obj.parent.sampler(args{:});
      else
        res = obj.parent.sampler(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function out = master(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.master);
    end
    function out = t(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.t);
    end
    function out = T(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.T);
    end
    function out = t0(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.t0);
    end
    function out = tf(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.tf);
    end
    function out = DT(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.DT);
    end
    function out = DT_control(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.DT_control);
    end
    function out = objective(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.objective);
    end
    function out = x(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.x);
    end
    function out = xq(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.xq);
    end
    function out = u(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.u);
    end
    function out = z(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.z);
    end
    function out = p(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.p);
    end
    function out = v(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.v);
    end
    function out = nx(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.nx);
    end
    function out = nz(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.nz);
    end
    function out = nu(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.nu);
    end
    function out = np(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.np);
    end
    function out = nv(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.nv);
    end
    function out = gist(obj)
      % Obtain an expression packing all information needed to obtain value/sample
      % 
      %         The composition of this array may vary between rockit versions
      % 
      %         Returns
      %         -------
      %         :obj:`~casadi.MX` column vector
      % 
      %         
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.gist);
    end
    function out = is_transcribed(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.is_transcribed);
    end
  end
end
