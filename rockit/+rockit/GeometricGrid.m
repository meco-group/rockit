classdef GeometricGrid < handle
  % Specify a geometrically growing grid
  %     
  %     Each control interval is a constant factor larger than its predecessor
  % 
  %     Parameters
  %     ----------
  %     growth_factor : float>1
  %         See `local` for interpretation
  %     local : bool, optional
  %         if False, last interval is growth_factor larger than first
  %         if True, interval k+1 is growth_factor larger than interval k
  %         Default: False
  %     min : float or  :obj:`casadi.MX`, optional
  %         Minimum size of control interval
  %         Enforced with constraints
  %         Default: 0
  %     max : float or  :obj:`casadi.MX`, optional
  %         Maximum size of control interval
  %         Enforced with constraints
  %         Default: inf
  % 
  %     Examples
  %     --------
  % 
  %     >>> MultipleShooting(N=3, grid=GeometricGrid(2)) # grid = [0, 1, 3, 7]*T/7
  %     >>> MultipleShooting(N=3, grid=GeometricGrid(2,local=True)) # grid = [0, 1, 5, 21]*T/21
  %     
  properties
    parent
  end
  methods
    function obj = GeometricGrid(varargin)
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.sampling_method.GeometricGrid')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'growth_factor','local','kwargs'});
      if isempty(kwargs)
        obj.parent = py.rockit.GeometricGrid(args{:});
      else
        obj.parent = py.rockit.GeometricGrid(args{:},pyargs(kwargs{:}));
      end
    end
    function varargout = subsref(obj,S)
      if ~strcmp(S(1).type,'()')
        [varargout{1:nargout}] = builtin('subsref',obj,S);
        return
      end
      varargin = S(1).subs;
      callee = py.getattr(obj.parent,'__call__');
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,3,{'t0','T','N'});
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
    function varargout = growth_factor(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'N'});
      if isempty(kwargs)
        res = obj.parent.growth_factor(args{:});
      else
        res = obj.parent.growth_factor(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = constrain_T(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,3,{'T','Tnext','N'});
      if isempty(kwargs)
        res = obj.parent.constrain_T(args{:});
      else
        res = obj.parent.constrain_T(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = scale_first(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'N'});
      if isempty(kwargs)
        res = obj.parent.scale_first(args{:});
      else
        res = obj.parent.scale_first(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = bounds_T(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,5,{'T_local','t0_local','k','T','N'});
      if isempty(kwargs)
        res = obj.parent.bounds_T(args{:});
      else
        res = obj.parent.bounds_T(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = normalized(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'N'});
      if isempty(kwargs)
        res = obj.parent.normalized(args{:});
      else
        res = obj.parent.normalized(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
  end
end
