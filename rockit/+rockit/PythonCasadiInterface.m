classdef PythonCasadiInterface < handle
    properties
        matlab_serializer
        matlab_deserializer
        python_serializer
        python_deserializer
    end
    methods
        function obj = PythonCasadiInterface()
            obj.matlab_serializer = casadi.StringSerializer();
            obj.matlab_serializer.pack(1);
            obj.matlab_deserializer = casadi.StringDeserializer(obj.matlab_serializer.encode());
            obj.matlab_deserializer.unpack();
            obj.python_serializer = py.casadi.StringSerializer();
            obj.python_serializer.pack(1);
            obj.python_deserializer = py.casadi.StringDeserializer(obj.python_serializer.encode());
            obj.python_deserializer.unpack();
            obj.python_serializer.connect(obj.python_deserializer);
            obj.python_deserializer.connect(obj.python_serializer);
            obj.matlab_serializer.connect(obj.matlab_deserializer);
            obj.matlab_deserializer.connect(obj.matlab_serializer);
        end
        function out = python2matlab_ret(obj, e)
          if isa(e,'py.tuple')
            for i=1:length(e)
               out{i} = obj.python2matlab(e{i});
            end
          elseif isa(e,'py.NoneType')
            out = {};
          else
            out{1} = obj.python2matlab(e);
          end
        end
        function [out,keywords] = matlab2python_arg(obj, v,n_in_min,arg_names)
          greedy_kwargs_from = inf;
          if n_in_min==-inf
            n_in_min = inf;
            greedy_kwargs_from = 1;
            for greedy_kwargs_from=1:numel(arg_names)
               if strcmp(arg_names{greedy_kwargs_from},'args')
                   break
               end
            end
          end
          assert(length(v)>=n_in_min || isinf(n_in_min))
          keywords_only = false;
          i_v = 1;
          i_arg = 1;
          out = {};
          keywords = {};
          has_kwargs = false;
          if length(arg_names)>0
              has_kwargs = strcmp(arg_names{end},'kwargs');
          end
          while i_v<=length(v)
            keywords_only = keywords_only || i_v>n_in_min;
            key_or_value = v{i_v};
            isc = ischar(key_or_value);
            if i_v>=greedy_kwargs_from && isc
                keywords_only = true;
                has_kwargs = true;
            end
            is_kwargs = strcmp(arg_names{i_arg},'kwargs');
            if ~keywords_only && is_kwargs
              keywords_only = true;
              i_v = i_v+1;
            end
            if ~keywords_only && isc && strcmp(key_or_value,arg_names{i_arg})
              keywords_only = true;
            end
            if ~keywords_only
                out{end+1} = obj.matlab2python(key_or_value);
            elseif isc && (ismember(key_or_value,arg_names(i_arg:end)) || has_kwargs)
              keywords_only = true;
              keywords{end+1} = key_or_value;
              keywords{end+1} = obj.matlab2python(v{i_v+1});
              i_v = i_v+1;
            else
              keywords{end+1} = arg_names{i_arg};
              keywords{end+1} = obj.matlab2python(key_or_value);
            end
            if ~is_kwargs
                i_arg = i_arg +1;
            end
            i_v = i_v+1;
          end
        end
        function out = python2matlab(obj, e)
          if isa(e,'py.numpy.ndarray')
            data = double(py.array.array('d',py.numpy.nditer(e,pyargs('order','F'))));
            dim = obj.python2matlab(e.shape);
            if length(dim)==0
              dim = {1 1};
            elseif length(dim)==1
              dim = {1 dim{1}};
            end
            out = reshape(data,[dim{:}]);
          elseif isa(e,'py.str')
            out = char(e);
          elseif isa(e,'py.int')
            out = double(e);
          elseif isa(e,'py.tuple') || isa(e,'py.list')
            out = cell(1,length(e));
            for i=1:length(e)
              out{i} = obj.python2matlab(e{i});
            end
          elseif isa(e,'py.generator')
            out = obj.python2matlab(py.list(e));
          else
            mod = '';
            try
              mod = char(py.getattr(e,'__module__'));
            except
            end
            a = split(class(e),'.');
            h = str2func([a{1} '.' a{2} '.matlab_path']);
            has_matlab = false;
            try
                h();
                has_matlab = true;
            catch
            end
            if strcmp(mod,'casadi.casadi')
              obj.python_serializer.pack(e);
              obj.matlab_deserializer.decode(char(obj.python_serializer.encode()));
              out = obj.matlab_deserializer.unpack();
            elseif has_matlab
              name = [a{2} '.' a{end}];
              h = str2func(name);
              try
                out = h(e);
              catch
                out = e;
              end
            elseif py.hasattr(e,'__call__')
              out = @(varargin) obj.apply(e,varargin{:});
            else
              out = e;
            end
          end
        end
        function out = matlab2python(obj, e)
          if isobject(e)
              m = metaclass(e);
              package = m.ContainingPackage.Name;
              if strcmp(package,'casadi')
                obj.matlab_serializer.pack(e);
                obj.python_deserializer.decode(obj.matlab_serializer.encode());
                out = obj.python_deserializer.unpack();
              elseif isprop(e,'parent')
                out = e.parent;
              else
                out = e;
              end
          elseif isnumeric(e)
              if isscalar(e) && floor(e)==e
                out = py.int(e);
              elseif isscalar(e)
                out = e;
              else
                out = py.numpy.array(py.memoryview(e(:)'));
              end
          elseif isstruct(e)
              names = fieldnames(e);
              d = {};
              for i=1:length(names)
                d{end+1} = names{i};
                d{end+1} = obj.matlab2python(e.(names{i}));
              end
              out = py.dict(pyargs(d{:}));
          elseif iscell(e)
              d = {};
              for i=1:length(e)
                d{end+1} = obj.matlab2python(e{i});
              end
              out = py.list(d);
          else
              out = e;
          end
        end
        function varargout = apply(obj,fun,varargin)
          global pythoncasadiinterface
          [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,1,{'t'});
          if isempty(kwargs)
            res = fun(args{:});
          else
            res = fun(args{:},pyargs(kwargs{:}));
          end
          varargout = pythoncasadiinterface.python2matlab_ret(res);
        end
    end
end



