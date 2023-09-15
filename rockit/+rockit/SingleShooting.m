classdef SingleShooting < rockit.DirectMethod
  properties
  end
  methods
    function obj = SingleShooting(varargin)
      obj@rockit.DirectMethod('from_super');
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.single_shooting.SingleShooting')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'kwargs'});
      if isempty(kwargs)
        obj.parent = py.rockit.SingleShooting(args{:});
      else
        obj.parent = py.rockit.SingleShooting(args{:},pyargs(kwargs{:}));
      end
    end
    function varargout = add_parameter(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','opti'});
      if isempty(kwargs)
        res = obj.parent.add_parameter(args{:});
      else
        res = obj.parent.add_parameter(args{:},pyargs(kwargs{:}));
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
    function varargout = add_constraints(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,2,{'stage','opti'});
      if isempty(kwargs)
        res = obj.parent.add_constraints(args{:});
      else
        res = obj.parent.add_constraints(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
  end
end
