import inspect
import os

import rockit

class MatlabEmittor:
  def __init__(self,package_name):
    self.python_ref_name = "parent"
    self.package_name = package_name
    self.dir = os.path.join("..","rockit","+"+self.package_name)
    os.makedirs(self.dir,exist_ok=True)

  def make_class(self,name,module,doc,inherit_from=None):
    return MatlabEmittorClass(self,name,module,doc,inherit_from=inherit_from)
    
  def to_matlab(self,e):
    if isinstance(e,list):
      return "{" + ",".join(self.to_matlab(ee) for ee in e) + "}"
    elif isinstance(e,str):
      return "'" + e.replace("'","''") + "'"
    elif e is None:
      return "[]"
    else:
      return str(e)

  def parse_signature(self,sig):
      arg_names = [k for k,v in sig.parameters.items()][1:]
      arg_defaults = [v.default for k,v in sig.parameters.items()][1:]
      n_in_min = 0
      for name,default in zip(arg_names,arg_defaults):
        if default is inspect._empty and sig.parameters[name].kind!= inspect.Parameter.VAR_KEYWORD:
          n_in_min += 1
        else:
          break
        if sig.parameters[name].kind==inspect.Parameter.VAR_POSITIONAL:
          n_in_min = 'inf'
          break
      if arg_names[-2:] == ["args","kwargs"]:
        raise Exception("Not supported")
      return str(n_in_min)+","+self.to_matlab(arg_names)
    
class MatlabEmittorClass:
  def __init__(self,parent,name,module,doc,inherit_from=None):
    self.name = name
    self.parent = parent
    self.module = module
    self.doc = doc
    self.out = open(os.path.join(self.parent.dir,name+".m"),"w")
    self.methods = []
    self.properties = []
    self.inherit_from = [] if inherit_from is None else inherit_from
    
  @property
  def ref(self):
    return self.parent.python_ref_name

  def add_method(self,name,signature,doc):
    self.methods.append((name,signature,doc))

  def add_property(self,name,doc):
    self.properties.append((name,doc))  

  def add_constructor(self,signature,doc):
    self.constructor = signature
    self.constructor_doc = doc

  def multi_line(self, prefix, arg, sig=None):
    if sig is not None:
      sig = "Arguments: " + ", ".join([k if v.default is inspect._empty else k+"="+str(v.default) for k,v in sig.parameters.items()][1:])
    if arg is None:
      return ""
    s = ""
    lines = arg.split("\n")
    if sig is not None:
      if len(lines)==0:
        lines = [sig]
      else:
        lines = [lines[0],sig]+lines[1:]

    for l in lines:
      s += prefix + "% " + l + "\n"
    return s

  def write(self):
    if len(self.inherit_from)==0:
      self.out.write("classdef {name} < handle\n".format(name=self.name))
    else:
      names = " & ".join(self.parent.package_name + "." + e.__name__ for e in self.inherit_from)
      self.out.write("classdef {name} < {names}\n".format(name=self.name,names=names))
    self.out.write(self.multi_line("  ", self.doc))
    self.out.write("  properties\n")
    if len(self.inherit_from)==0:
      self.out.write("    {name}\n".format(name=self.ref))

    self.out.write("  end\n")

    self.out.write("  methods\n")
    
    self.out.write("    function obj = {name}(varargin)\n".format(name=self.name))
    self.out.write(self.multi_line("      ", self.constructor_doc, self.constructor))
    for e in self.inherit_from:
      self.out.write("      obj@{super_name}('from_super');\n".format(super_name=self.parent.package_name+"."+e.__name__))
    self.out.write("      if length(varargin)==1 && ischar(varargin{1}) && strcmp(varargin{1},'from_super'),return,end\n")
    self.out.write("      if length(varargin)==1 && isa(varargin{{1}},'py.{module}.{name}')\n".format(name=self.name,module=self.module))
    self.out.write("        obj.{ref} = varargin{{1}};\n".format(ref=self.ref));
    self.out.write("        return\n");
    self.out.write("      end\n");
    self.out.write("      global pythoncasadiinterface\n");
    self.out.write("      if isempty(pythoncasadiinterface)\n");
    self.out.write("        pythoncasadiinterface = {package_name}.PythonCasadiInterface;\n".format(package_name=self.parent.package_name));
    self.out.write("      end\n");
    self.out.write("      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,{sig});\n".format(sig=self.parent.parse_signature(self.constructor)))
    self.out.write("      if isempty(kwargs)\n")
    self.out.write("        obj.{ref} = py.rockit.{name}(args{{:}});\n".format(ref=self.ref,name=self.name))
    self.out.write("      else\n")
    self.out.write("        obj.{ref} = py.rockit.{name}(args{{:}},pyargs(kwargs{{:}}));\n".format(ref=self.ref,name=self.name))
    self.out.write("      end\n")
    self.out.write("    end\n")
    for m,sig,doc in self.methods:
      if m.startswith("spy_"):
        continue
      if m=="spy":
        self.out.write("    function out = spy(obj)\n".format(name=m))
        self.out.write("      figure;\n")
        self.out.write("      subplot(1,2,1);\n")
        self.out.write("      [J, titleJ] = obj.jacobian(true);\n");
        self.out.write("      spy(casadi.DM(J));\n")
        self.out.write("      title(titleJ);\n");
        self.out.write("      subplot(1,2,2);\n")
        self.out.write("      [H, titleH] = obj.hessian(true);\n");
        self.out.write("      spy(casadi.DM(H));\n")
        self.out.write("      title(titleH);\n");
        self.out.write("    end\n")  
        continue      
      if m=="__call__":
        self.out.write("    function varargout = subsref(obj,S)\n".format(name=m))
        self.out.write("      if ~strcmp(S(1).type,'()')\n")
        self.out.write("        [varargout{1:nargout}] = builtin('subsref',obj,S);\n")
        self.out.write("        return\n")
        self.out.write("      end\n")
        self.out.write("      varargin = S(1).subs;\n")
        self.out.write("      callee = py.getattr(obj.parent,'__call__');\n")
      else:
        self.out.write("    function varargout = {name}(obj,varargin)\n".format(name=m))
      self.out.write(self.multi_line("      ", doc, sig))
      self.out.write("      global pythoncasadiinterface\n");
      self.out.write("      [args,kwargs] = pythoncasadiinterface.matlab2python_arg(varargin,{sig});\n".format(sig=self.parent.parse_signature(sig)))
      if m in ["subject_to","state","control","variable","parameter","algebraic"]:
        self.out.write("      meta = py.None;\n")
        self.out.write("      try\n")
        self.out.write("        st = dbstack('-completenames',1);\n")
        self.out.write("        if length(st)>0\n")
        self.out.write("          meta = struct('stacktrace', {{st(1)}});\n")
        self.out.write("          meta = pythoncasadiinterface.matlab2python(meta);\n")
        self.out.write("        end\n")
        self.out.write("      catch\n")
        self.out.write("      end\n")
        self.out.write("      kwargs = {kwargs{:} 'meta' meta};\n")
      if m=="__call__":
        callee = "callee"
      else:
        callee = "obj.{ref}.{name}".format(name=m,ref=self.ref)
      self.out.write("      if isempty(kwargs)\n")
      self.out.write("        res = {callee}(args{{:}});\n".format(callee=callee))
      self.out.write("      else\n")
      self.out.write("        res = {callee}(args{{:}},pyargs(kwargs{{:}}));\n".format(callee=callee))
      self.out.write("      end\n")
      self.out.write("      varargout = pythoncasadiinterface.python2matlab_ret(res);\n")
      if m=="__call__":
        self.out.write("       if (length(S)>1) && strcmp(S(2).type,'.')\n")
        self.out.write("         res = varargout{1};\n")
        self.out.write("         [varargout{1:nargout}] = builtin('subsref',res,S(2:end));\n")
        self.out.write("       end\n")

      self.out.write("    end\n")
    for m,doc in self.properties:
      self.out.write("    function out = {name}(obj)\n".format(name=m))
      self.out.write(self.multi_line("      ", doc))
      self.out.write("      global pythoncasadiinterface\n");
      self.out.write("      out = pythoncasadiinterface.python2matlab(obj.{ref}.{name});\n".format(name=m,ref=self.ref))
      self.out.write("    end\n")
    self.out.write("  end\n")

    self.out.write("end\n")
  
me = MatlabEmittor("rockit")
for class_name, cl in rockit.__dict__.items():
  
  if isinstance(cl, type):
    subclasses = cl.mro()[1:-1]
    exposed_subclasses = [e for e in subclasses if e in rockit.__dict__.values()]
    print(exposed_subclasses)
    ce = me.make_class(class_name,inspect.getmodule(cl).__name__,cl.__doc__,inherit_from=exposed_subclasses)

    for method_name, method in  cl.__dict__.items():
      #print(method_name, method, inspect.ismethod(method),inspect.isfunction(method))
      if inspect.isfunction(method):
        if method.__name__=="__init__":
          ce.add_constructor(inspect.signature(method),method.__doc__)
        elif method.__name__.startswith("_") and method.__name__!="__call__":
          continue
        else:
          if hasattr(method,'_decorator_original'):
            method = method._decorator_original
          ce.add_method(method_name,inspect.signature(method),method.__doc__)
      else:
          if method_name.startswith("_"): continue
          print(method_name)

          ce.add_property(method_name, method.__doc__)

    ce.write()
