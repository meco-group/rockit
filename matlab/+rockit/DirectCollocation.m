classdef DirectCollocation < rockit.DirectMethod
  properties
  end
  methods
    function obj = DirectCollocation(varargin)
      obj@rockit.DirectMethod('from_super');
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.direct_collocation.DirectCollocation')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'degree','scheme','kwargs'});
      if isempty(kwargs)
        obj.parent = py.rockit.DirectCollocation(args{:});
      else
        obj.parent = py.rockit.DirectCollocation(args{:},pyargs(kwargs{:}));
      end
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
    function varargout = set_initial(obj,varargin)
      global pythoncasadiinterface
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,3,{'stage','opti','initial'});
      if isempty(kwargs)
        res = obj.parent.set_initial(args{:});
      else
        res = obj.parent.set_initial(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
  end
end