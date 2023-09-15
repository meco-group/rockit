classdef DirectMethod < handle
  % 
  %     Base class for 'direct' solution methods for Optimal Control Problems:
  %       'first discretize, then optimize'
  %     
  properties
    parent
  end
  methods
    function obj = DirectMethod(varargin)
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.direct_method.DirectMethod')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        obj.parent = py.rockit.DirectMethod(args{:});
      else
        obj.parent = py.rockit.DirectMethod(args{:},pyargs(kwargs{:}));
      end
    end
    function varargout = clean(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = obj.parent.clean(args{:});
      else
        res = obj.parent.clean(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
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
    function varargout = inherit(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'template'});
      if isempty(kwargs)
        res = obj.parent.inherit(args{:});
      else
        res = obj.parent.inherit(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = eval(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','expr'});
      if isempty(kwargs)
        res = obj.parent.eval(args{:});
      else
        res = obj.parent.eval(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = eval_top(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','expr'});
      if isempty(kwargs)
        res = obj.parent.eval_top(args{:});
      else
        res = obj.parent.eval_top(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = add_variables(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','opti'});
      if isempty(kwargs)
        res = obj.parent.add_variables(args{:});
      else
        res = obj.parent.add_variables(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = add_parameters(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','opti'});
      if isempty(kwargs)
        res = obj.parent.add_parameters(args{:});
      else
        res = obj.parent.add_parameters(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = untranscribe(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage','phase','kwargs'});
      if isempty(kwargs)
        res = obj.parent.untranscribe(args{:});
      else
        res = obj.parent.untranscribe(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = untranscribe_placeholders(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'phase','stage'});
      if isempty(kwargs)
        res = obj.parent.untranscribe_placeholders(args{:});
      else
        res = obj.parent.untranscribe_placeholders(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = main_untranscribe(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage','phase','kwargs'});
      if isempty(kwargs)
        res = obj.parent.main_untranscribe(args{:});
      else
        res = obj.parent.main_untranscribe(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = main_transcribe(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage','phase','kwargs'});
      if isempty(kwargs)
        res = obj.parent.main_transcribe(args{:});
      else
        res = obj.parent.main_transcribe(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = transcribe(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage','phase','kwargs'});
      if isempty(kwargs)
        res = obj.parent.transcribe(args{:});
      else
        res = obj.parent.transcribe(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_initial(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,3,{'stage','master','initial'});
      if isempty(kwargs)
        res = obj.parent.set_initial(args{:});
      else
        res = obj.parent.set_initial(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_parameter(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','opti'});
      if isempty(kwargs)
        res = obj.parent.set_parameter(args{:});
      else
        res = obj.parent.set_parameter(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_value(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,4,{'stage','master','parameter','value'});
      if isempty(kwargs)
        res = obj.parent.set_value(args{:});
      else
        res = obj.parent.set_value(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = transcribe_placeholders(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,3,{'phase','stage','placeholders'});
      if isempty(kwargs)
        res = obj.parent.transcribe_placeholders(args{:});
      else
        res = obj.parent.transcribe_placeholders(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = non_converged_solution(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage'});
      if isempty(kwargs)
        res = obj.parent.non_converged_solution(args{:});
      else
        res = obj.parent.non_converged_solution(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = solve(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage'});
      if isempty(kwargs)
        res = obj.parent.solve(args{:});
      else
        res = obj.parent.solve(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = solve_limited(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage'});
      if isempty(kwargs)
        res = obj.parent.solve_limited(args{:});
      else
        res = obj.parent.solve_limited(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = callback(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','fun'});
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
    function varargout = initial_value(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','expr'});
      if isempty(kwargs)
        res = obj.parent.initial_value(args{:});
      else
        res = obj.parent.initial_value(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = to_function(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'stage','name','args','results','margs'});
      if isempty(kwargs)
        res = obj.parent.to_function(args{:});
      else
        res = obj.parent.to_function(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = fill_placeholders_integral(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'phase','stage','expr','args'});
      if isempty(kwargs)
        res = obj.parent.fill_placeholders_integral(args{:});
      else
        res = obj.parent.fill_placeholders_integral(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = fill_placeholders_T(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'phase','stage','expr','args'});
      if isempty(kwargs)
        res = obj.parent.fill_placeholders_T(args{:});
      else
        res = obj.parent.fill_placeholders_T(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = fill_placeholders_t0(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'phase','stage','expr','args'});
      if isempty(kwargs)
        res = obj.parent.fill_placeholders_t0(args{:});
      else
        res = obj.parent.fill_placeholders_t0(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = fill_placeholders_t(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'phase','stage','expr','args'});
      if isempty(kwargs)
        res = obj.parent.fill_placeholders_t(args{:});
      else
        res = obj.parent.fill_placeholders_t(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = fill_placeholders_DT(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'phase','stage','expr','args'});
      if isempty(kwargs)
        res = obj.parent.fill_placeholders_DT(args{:});
      else
        res = obj.parent.fill_placeholders_DT(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = fill_placeholders_DT_control(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'phase','stage','expr','args'});
      if isempty(kwargs)
        res = obj.parent.fill_placeholders_DT_control(args{:});
      else
        res = obj.parent.fill_placeholders_DT_control(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
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
end
