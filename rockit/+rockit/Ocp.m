classdef Ocp < rockit.Stage
  properties
  end
  methods
    function obj = Ocp(varargin)
      % Create an Optimal Control Problem environment
      % Arguments: t0=0, T=1, kwargs
      % 
      %         Parameters
      %         ----------
      %         t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Starting time of the optimal control horizon
      %             Default: 0
      %         T : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Total horizon of the optimal control horizon
      %             Default: 1
      % 
      %         Examples
      %         --------
      % 
      %         >>> ocp = Ocp()
      %         
      obj@rockit.Stage('from_super');
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.ocp.Ocp')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'t0','T','kwargs'});
      if isempty(kwargs)
        obj.parent = py.rockit.Ocp(args{:});
      else
        obj.parent = py.rockit.Ocp(args{:},pyargs(kwargs{:}));
      end
    end
    function varargout = jacobian(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'with_label'});
      if isempty(kwargs)
        res = obj.parent.jacobian(args{:});
      else
        res = obj.parent.jacobian(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = hessian(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'with_label'});
      if isempty(kwargs)
        res = obj.parent.hessian(args{:});
      else
        res = obj.parent.hessian(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function out = spy(obj)
      figure;
      subplot(1,2,1);
      [J, titleJ] = obj.jacobian(true);
      spy(casadi.DM(J));
      title(titleJ);
      subplot(1,2,2);
      [H, titleH] = obj.hessian(true);
      spy(casadi.DM(H));
      title(titleH);
    end
    function varargout = internal_transcribe(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.internal_transcribe(args{:});
      else
        res = obj.parent.internal_transcribe(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = internal_untranscribe(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.internal_untranscribe(args{:});
      else
        res = obj.parent.internal_untranscribe(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = solve(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.solve(args{:});
      else
        res = obj.parent.solve(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = solve_limited(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.solve_limited(args{:});
      else
        res = obj.parent.solve_limited(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = callback(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'fun'});
      if isempty(kwargs)
        res = obj.parent.callback(args{:});
      else
        res = obj.parent.callback(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = solver(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'solver','solver_options'});
      if isempty(kwargs)
        res = obj.parent.solver(args{:});
      else
        res = obj.parent.solver(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = show_infeasibilities(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'args'});
      if isempty(kwargs)
        res = obj.parent.show_infeasibilities(args{:});
      else
        res = obj.parent.show_infeasibilities(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = debugme(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'e'});
      if isempty(kwargs)
        res = obj.parent.debugme(args{:});
      else
        res = obj.parent.debugme(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = to_function(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'name','args','results','margs'});
      if isempty(kwargs)
        res = obj.parent.to_function(args{:});
      else
        res = obj.parent.to_function(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = is_sys_time_varying(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.is_sys_time_varying(args{:});
      else
        res = obj.parent.is_sys_time_varying(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = is_parameter_appearing_in_sys(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.is_parameter_appearing_in_sys(args{:});
      else
        res = obj.parent.is_parameter_appearing_in_sys(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = sys_dae(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.sys_dae(args{:});
      else
        res = obj.parent.sys_dae(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = view_api(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'name'});
      if isempty(kwargs)
        res = obj.parent.view_api(args{:});
      else
        res = obj.parent.view_api(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = sys_simulator(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'intg','intg_options'});
      if isempty(kwargs)
        res = obj.parent.sys_simulator(args{:});
      else
        res = obj.parent.sys_simulator(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = save(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'name'});
      if isempty(kwargs)
        res = obj.parent.save(args{:});
      else
        res = obj.parent.save(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function out = internal__module__(obj)
      % str(object='') -> str
      % str(bytes_or_buffer[, encoding[, errors]]) -> str
      % 
      % Create a new string object from the given object. If encoding or
      % errors is specified, then the object must expose a data buffer
      % that will be decoded using the given encoding and error handler.
      % Otherwise, returns the result of object.__str__() (if defined)
      % or repr(object).
      % encoding defaults to sys.getdefaultencoding().
      % errors defaults to 'strict'.
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.internal__module__);
    end
    function out = internal_transcribed(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.internal_transcribed);
    end
    function out = placeholders_transcribed(obj)
      % 
      %         May also be called after solving (issue #91)
      %         
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.placeholders_transcribed);
    end
    function out = non_converged_solution(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.non_converged_solution);
    end
    function out = debug(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.debug);
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
    function out = load(obj)
      % staticmethod(function) -> method
      % 
      % Convert a function to be a static method.
      % 
      % A static method does not receive an implicit first argument.
      % To declare a static method, use this idiom:
      % 
      %      class C:
      %          @staticmethod
      %          def f(arg1, arg2, ...):
      %              ...
      % 
      % It can be called either on the class (e.g. C.f()) or on an instance
      % (e.g. C().f()).  The instance is ignored except for its class.
      % 
      % Static methods in Python are similar to those found in Java or C++.
      % For a more advanced concept, see the classmethod builtin.
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.load);
    end
    function out = internal__doc__(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.internal__doc__);
    end
  end
end
