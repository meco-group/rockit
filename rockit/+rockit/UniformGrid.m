classdef UniformGrid < handle
  % Specify a grid with uniform spacing
  %     
  %     Parameters
  %     ----------
  %     min : float or  :obj:`casadi.MX`, optional
  %         Minimum size of control interval
  %         Enforced with constraints
  %         Default: 0
  %     max : float or  :obj:`casadi.MX`, optional
  %         Maximum size of control interval
  %         Enforced with constraints
  %         Default: inf
  %     
  properties
    parent
  end
  methods
    function obj = UniformGrid(varargin)
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.sampling_method.UniformGrid')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'kwargs'});
      if isempty(kwargs)
        obj.parent = py.rockit.UniformGrid(args{:});
      else
        obj.parent = py.rockit.UniformGrid(args{:},pyargs(kwargs{:}));
      end
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
