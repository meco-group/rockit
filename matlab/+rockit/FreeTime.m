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
  end
end
