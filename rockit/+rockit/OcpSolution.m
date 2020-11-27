classdef OcpSolution < handle
  properties
    parent
  end
  methods
    function obj = OcpSolution(varargin)
      % Wrap casadi.nlpsol to simplify access to numerical solution.
      % Arguments: nlpsol, stage
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.solution.OcpSolution')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'nlpsol','stage'});
      if isempty(kwargs)
        obj.parent = py.rockit.OcpSolution(args{:});
      else
        obj.parent = py.rockit.OcpSolution(args{:},pyargs(kwargs{:}));
      end
    end
    function varargout = subsref(obj,S)
      if ~strcmp(S(1).type,'()')
        [varargout{1:nargout}] = builtin('subsref',obj,S);
        return
      end
      varargin = S(1).subs;
      callee = py.getattr(obj.parent,'__call__');
      % Sample expression at solution on a given grid.
      % Arguments: stage
      % 
      %         Parameters
      %         ----------
      %         stage : :obj:`~rockit.stage.Stage`
      %             An optimal control problem stage.
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'stage'});
      if isempty(kwargs)
        res = callee(args{:});
      else
        res = callee(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
       if (length(S)>1) && strcmp(S(2).type,'.')
         res = varargout{1};
         [varargout{1:nargout}] = builtin('subsref',res,S(2:end));
       end
    end
    function varargout = value(obj,varargin)
      % Get the value of an (non-signal) expression.
      % Arguments: expr, args
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`casadi.MX`
      %             Arbitrary expression containing no signals (states, controls) ...
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,inf,{'expr','args'});
      if isempty(kwargs)
        res = obj.parent.value(args{:});
      else
        res = obj.parent.value(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = sample(obj,varargin)
      % Sample expression at solution on a given grid.
      % Arguments: expr, grid, kwargs
      % 
      %         Parameters
      %         ----------
      %         expr : :obj:`casadi.MX`
      %             Arbitrary expression containing states, controls, ...
      %         grid : `str`
      %             At which points in time to sample, options are
      %             'control' or 'integrator' (at integrator discretization
      %             level) or 'integrator_roots'.
      %         refine : int, optional
      %             Refine grid by evaluation the polynomal of the integrater at
      %             intermediate points ("refine" points per interval).
      % 
      %         Returns
      %         -------
      %         time : numpy.ndarray
      %             Time from zero to final time, same length as res
      %         res : numpy.ndarray
      %             Numerical values of evaluated expression at points in time vector.
      % 
      %         Examples
      %         --------
      %         Assume an ocp with a stage is already defined.
      % 
      %         >>> sol = ocp.solve()
      %         >>> tx, xs = sol.sample(x, grid='control')
      %         
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'expr','grid','kwargs'});
      if isempty(kwargs)
        res = obj.parent.sample(args{:});
      else
        res = obj.parent.sample(args{:},pyargs(kwargs{:}));
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
      %         t -> output
      % 
      %         mode 1 : Python Function
      %             Symbolically evaluated expression at points in time vector.
      %         mode 2 : :obj:`casadi.Function`
      %             Time from zero to final time, same length as res
      % 
      %         Examples
      %         --------
      %         Assume an ocp with a stage is already defined.
      % 
      %         >>> sol = ocp.solve()
      %         >>> s = sol.sampler(x)
      %         >>> s(1.0) # Value of x at t=1.0
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
    function out = gist(obj)
      % All numerical information needed to compute any value/sample
      % 
      %         Returns
      %         -------
      %         1D numpy.ndarray
      %            The composition of this array may vary between rockit versions
      % 
      % 
      %         
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.gist);
    end
    function out = stats(obj)
      % Retrieve solver statistics
      % 
      %         Returns
      %         -------
      %         Dictionary
      %            The information contained is not structured and may change between rockit versions
      % 
      %         
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.stats);
    end
  end
end
