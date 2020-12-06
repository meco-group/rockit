function varargout = external_method(varargin)
  global pythoncasadiinterface
  [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'name','kwargs'});
  if isempty(kwargs)
    res = py.rockit.external_method(args{:});
  else
    res = py.rockit.external_method(args{:},pyargs(kwargs{:}));
  end
  varargout = pythoncasadiinterface.python2matlab_ret(res);
end
