classdef GlobalOptions < handle
  properties
    parent
  end
  methods
    function obj = GlobalOptions(varargin)
    end
  end
  methods(Static)
    function varargout = set_cmake_flags(varargin)
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'flags'});
      if isempty(kwargs)
        res = py.rockit.global_options.GlobalOptions.set_cmake_flags(args{:});
      else
        res = py.rockit.global_options.GlobalOptions.set_cmake_flags(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = get_cmake_flags(varargin)
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = py.rockit.global_options.GlobalOptions.get_cmake_flags(args{:});
      else
        res = py.rockit.global_options.GlobalOptions.get_cmake_flags(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_compiler(varargin)
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'arg'});
      if isempty(kwargs)
        res = py.rockit.global_options.GlobalOptions.set_compiler(args{:});
      else
        res = py.rockit.global_options.GlobalOptions.set_compiler(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = get_compiler(varargin)
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = py.rockit.global_options.GlobalOptions.get_compiler(args{:});
      else
        res = py.rockit.global_options.GlobalOptions.get_compiler(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = set_cmake_build_type(varargin)
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'arg'});
      if isempty(kwargs)
        res = py.rockit.global_options.GlobalOptions.set_cmake_build_type(args{:});
      else
        res = py.rockit.global_options.GlobalOptions.set_cmake_build_type(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
    function varargout = get_cmake_build_type(varargin)
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{});
      if isempty(kwargs)
        res = py.rockit.global_options.GlobalOptions.get_cmake_build_type(args{:});
      else
        res = py.rockit.global_options.GlobalOptions.get_cmake_build_type(args{:},pyargs(kwargs{:}));
      end
      varargout = pythoncasadiinterface.python2matlab_ret(res);
    end
  end
end
