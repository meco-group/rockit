function varargout = interface_simulink(varargin)
  global pythoncasadiinterface
    if isempty(pythoncasadiinterface)
    pythoncasadiinterface = impact.PythonCasadiInterface;
  end
  [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'mdl','path','block','exclude_outputs','exclude_inputs','options','jacobian_options'});
  if isempty(kwargs)
    res = py.rockit.interface_simulink(args{:});
  else
    res = py.rockit.interface_simulink(args{:},pyargs(kwargs{:}));
  end
  varargout = pythoncasadiinterface.python2matlab_ret(res);
end
