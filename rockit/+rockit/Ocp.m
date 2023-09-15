classdef Ocp < rockit.Stage
  properties
  end
  methods
    function obj = Ocp(varargin)
      % Create an Optimal Control Problem environment
      % Arguments: t0=0, T=1, scale=1, kwargs
      % 
      %         Parameters
      %         ----------
      %         t0 : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Starting time of the optimal control horizon
      %             Default: 0
      %         T : float or :obj:`~rockit.freetime.FreeTime`, optional
      %             Total horizon of the optimal control horizon
      %             Default: 1
      %         scale: float, optional
      %                Typical time scale
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
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'t0','T','scale','kwargs'});
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
    function varargout = transcribe(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'kwargs'});
      if isempty(kwargs)
        res = obj.parent.transcribe(args{:});
      else
        res = obj.parent.transcribe(args{:},pyargs(kwargs{:}));
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
  end
  methods(Static)
    function varargout = load(varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'name'});
      if isempty(kwargs)
        res = py.rockit.ocp.Ocp.load(args{:});
      else
        res = py.rockit.ocp.Ocp.load(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
  end
end
