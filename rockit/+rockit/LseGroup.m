classdef LseGroup < handle
  properties
    parent
  end
  methods
    function obj = LseGroup(varargin)
      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end
      if length(varargin)==1 && isa(varargin{1},'py.rockit.grouping_techniques.LseGroup')
        obj.parent = varargin{1};
        return
      end
      global pythoncasadiinterface
      if isempty(pythoncasadiinterface)
        pythoncasadiinterface = rockit.PythonCasadiInterface;
      end
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,0,{'margin_abs'});
      if isempty(kwargs)
        obj.parent = py.rockit.LseGroup(args{:});
      else
        obj.parent = py.rockit.LseGroup(args{:},pyargs(kwargs{:}));
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
      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'M','axis'});
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
    function out = class_id(obj)
      % int([x]) -> integer
      % int(x, base=10) -> integer
      % 
      % Convert a number or string to an integer, or return 0 if no arguments
      % are given.  If x is a number, return x.__int__().  For floating point
      % numbers, this truncates towards zero.
      % 
      % If x is not a number or if base is given, then x must be a string,
      % bytes, or bytearray instance representing an integer literal in the
      % given base.  The literal can be preceded by '+' or '-' and be surrounded
      % by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
      % Base 0 means to interpret the base from the string as an integer literal.
      % >>> int('0b100', base=0)
      % 4
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.class_id);
    end
    function out = tuple(obj)
      global pythoncasadiinterface
      out = pythoncasadiinterface.python2matlab(obj.parent.tuple);
    end
  end
end
