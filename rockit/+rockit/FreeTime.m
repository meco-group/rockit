classdef FreeTime < handle
  properties
    parent
  end
  methods
    function obj = FreeTime(varargin)
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.freetime.FreeTime')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'T_init'});
      if isempty(kwargs)
        obj.parent = py.rockit.FreeTime(args{:});
      else
        obj.parent = py.rockit.FreeTime(args{:},pyargs(kwargs{:}));
      end
    end
    function out = internal__module__(obj)
      % str(object='') -> str
      % str(bytes_or_buffer[, encoding[, errors]]) -> str
      % 
      % Create a new string object from the given object. If encoding or
      % errors is specified, then the object must expose a data buffer
      % that will be decoded using the given encoding and error handler.
      % Otherwise, returns the result of object.__str__() (if defined)
      % or repr(object).
      % encoding defaults to sys.getdefaultencoding().
      % errors defaults to 'strict'.
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.internal__module__);
    end
    function out = internal__dict__(obj)
      % dictionary for instance variables (if defined)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.internal__dict__);
    end
    function out = internal__weakref__(obj)
      % list of weak references to the object (if defined)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.internal__weakref__);
    end
    function out = internal__doc__(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.internal__doc__);
    end
  end
end
