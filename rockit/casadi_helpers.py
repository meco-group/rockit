#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

import casadi as cs
from casadi import *
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

def get_ranges_dict(list_expr):
    ret = HashDict()
    offset = 0
    for e in list_expr:
        next_offset = offset+e.nnz()
        ret[e] = list(range(offset, next_offset))
        offset = next_offset
    return ret

def reinterpret_expr(expr, symbols_from, symbols_to):
    """
    .. code-block:: python
    
        x = MX.sym("x")
        y = MX.sym("y")

        z = -(x*y**2+2*x)**4<0

        print(reinterpret_expr(z,[y],[sin(y)]))
    """

    f = Function('f', symbols_from, [expr])

    # Work vector
    work = [None for i in range(f.sz_w())]

    output_val = [None]

    # Loop over the algorithm
    for k in range(f.n_instructions()):

        # Get the atomic operation
        op = f.instruction_id(k)

        o = f.instruction_output(k)
        i = f.instruction_input(k)

        if(op == OP_CONST):
            v = f.instruction_MX(k).to_DM()
            work[o[0]] = v
        else:
            if op == OP_INPUT:
                work[o[0]] = symbols_to[i[0]]
            elif op == OP_OUTPUT:
                output_val[o[0]] = work[i[0]]
            elif op == OP_ADD:
                work[o[0]] = work[i[0]] + work[i[1]]
            elif op == OP_TWICE:
                work[o[0]] = 2 * work[i[0]]
            elif op == OP_SUB:
                work[o[0]] = work[i[0]] - work[i[1]]
            elif op == OP_MUL:
                work[o[0]] = work[i[0]] * work[i[1]]
            elif op == OP_MTIMES:
                work[o[0]] = np.dot(work[i[1]], work[i[2]]) + work[i[0]]
            elif op == OP_PARAMETER:
                work[o[0]] = f.instruction_MX(k)
            elif op == OP_SQ:
                work[o[0]] = work[i[0]]**2
            elif op == OP_LE:
                work[o[0]] = work[i[0]] <= work[i[1]]
            elif op == OP_LT:
                work[o[0]] = work[i[0]] < work[i[1]]
            elif op == OP_NEG:
                work[o[0]] = -work[i[0]]
            elif op == OP_CONSTPOW:
                work[o[0]] = work[i[0]]**work[i[1]]
            else:
                print('Unknown operation: ', op)

                print('------')
                print('Evaluated ' + str(f))

    return output_val[0]


def get_meta(base=None):
    if base is not None: return base
    # Construct meta-data
    import sys
    import os
    try:
        frame = sys._getframe(2)
        meta = {"stacktrace": [{"file":os.path.abspath(frame.f_code.co_filename),"line":frame.f_lineno,"name":frame.f_code.co_name} ] }
    except:
        meta = {"stacktrace": []}
    return meta

def merge_meta(a, b):
    if b is None:
        return a
    if a is None:
        return b
    from copy import deepcopy
    res = deepcopy(a)
    res["stacktrace"] += b["stacktrace"] 
    return res

def single_stacktrace(m):
    from copy import deepcopy
    m = deepcopy(m)
    m["stacktrace"] = m["stacktrace"][0]
    return m

def reshape_number(target, value):
    if not isinstance(value,cs.MX):
        value = cs.DM(value)
    if value.is_scalar():
        value = cs.DM.ones(target.shape)*value
    return value

def is_numeric(expr):
    if isinstance(expr,cs.DM):
        return True
    elif isinstance(expr,np.ndarray):
        return True
    try:
        expr = evalf(expr)
        return True
    except:
        return False

def DM2numpy(dm, expr_shape, tdim=None):
    if tdim is None:
        return np.array(dm).squeeze()
    expr_prod = expr_shape[0]*expr_shape[1]

    target_shape = (tdim,)+tuple([e for e in expr_shape if e!=1])

    res = np.array(dm).reshape(expr_shape[0], tdim, expr_shape[1])
    res = np.transpose(res,[1,0,2])
    res = res.reshape(target_shape)
    return res

class HashWrap:
    def __init__(self, arg):
        assert not isinstance(arg,HashWrap)
        self.arg = arg

    def __hash__(self):
        return hash(self.arg)

    def __eq__(self, b):
        # key in dict
        b_arg = b
        if isinstance(b,HashWrap): b_arg = b.arg
        if self.arg is b_arg: return True
        r = self.arg==b_arg
        return r.is_one()

    def __str__(self):
        return self.arg.__str__()
    def __repr__(self):
        return self.arg.__repr__()

class HashDict(dict):
    def __init__(self,*args,**kwargs):
        r = dict(*args,**kwargs)
        dict.__init__(self)
        for k,v in r.items():
            self[k] = v
    def __getitem__(self, k):
        return dict.__getitem__(self, HashWrap(k))
    def __setitem__(self, k, v):
        return dict.__setitem__(self, HashWrap(k), v)
    def keys(self):
        for k in dict.keys(self):
            yield k.arg
    def items(self):
        for k,v in dict.items(self):
            yield k.arg, v
    __iter__ = keys
    def __copy__(self):
        r = HashDict()
        for k,v in self.items():
            r[k] = v
        return r

class HashList(list):
    def __init__(self,*args,**kwargs):
        r = list(*args,**kwargs)
        list.__init__(self)
        for v in r:
            self.append(v)
        self._stored = set()
    def append(self, item):
        list.append(self, item)
        self._stored.add(HashWrap(item))
    def __contains__(self, item):
        return item in self._stored
    def __copy__(self):
        r = HashList()
        for v in self:
            r.append(v)
        return r

class HashDefaultDict(defaultdict):
    def __init__(self,default_factory=None, *args,**kwargs):
        r = defaultdict(default_factory,*args,**kwargs)
        defaultdict.__init__(self)
        for k,v in r.items():
            self[k] = v
    def __getitem__(self, k):
        return defaultdict.__getitem__(self, HashWrap(k))
    def __setitem__(self, k, v):
        return defaultdict.__setitem__(self, HashWrap(k), v)
    def keys(self):
        ret = []
        for k in defaultdict.keys(self):
            ret.append(k.arg)
        return ret
    def items(self):
        for k,v in defaultdict.items(self):
            yield k.arg, v
    def __iter__(self):
        for k in defaultdict.__iter__(self):
            yield k.arg
    def __copy__(self):
        r = HashDefaultDict(self.default_factory)
        for k,v in self.items():
            r[k] = v
        return r

class HashOrderedDict(OrderedDict):
    def __init__(self,*args,**kwargs):
        r = OrderedDict(*args,**kwargs)
        OrderedDict.__init__(self)
        for k,v in r.items():
            self[k] = v
    def __getitem__(self, k):
        return OrderedDict.__getitem__(self, HashWrap(k))
    def __setitem__(self, k, v):
        return OrderedDict.__setitem__(self, HashWrap(k), v)
    def keys(self):
        ret = []
        for k in self:
            ret.append(k)
        return ret
    def items(self):
        for k in self:
            yield k, self[k]
    def __iter__(self):
        for k in OrderedDict.__iter__(self):
            yield k.arg
    def __copy__(self):
        r = HashOrderedDict()
        for k,v in self.items():
            r[k] = v
        return r


def for_all_primitives(expr, rhs, callback, msg, rhs_type=MX):
    if expr.is_symbolic():
        callback(expr, rhs)
        return
    if expr.is_valid_input():
        # Promote to correct size
        rhs = rhs_type(rhs)
        if rhs.is_scalar() and not expr.is_scalar():
            rhs = rhs*rhs_type(expr.sparsity())
        rhs = rhs[expr.sparsity()]
        rhs = vec(rhs)
        prim = expr.primitives()
        rhs = rhs.nz
        offset = 0
        for p in prim:
            callback(p, rhs_type(p.sparsity(), rhs[offset:offset+p.nnz()]))
            offset += p.nnz()
    else:
        raise Exception(msg)
        

class Node:
  def __init__(self,val):
    self.val = val
    self.nodes = []

class AutoBrancher:
  OPEN = 0
  DONE = 1
  def __init__(self):
    self.root = Node(AutoBrancher.OPEN)
    self.trace = [self.root]

  @property
  def current(self):
    return self.trace[-1]

  def branch(self, alternatives = [True, False]):
    alternatives = list(alternatives)
    nodes = self.current.nodes
    if len(nodes)==0:
      nodes += [None]*len(alternatives)
    for i,n in enumerate(nodes):
      if n is None:
        nodes[i] = Node(AutoBrancher.OPEN)
        self.trace.append(nodes[i])
        self.this_branch.append(alternatives[i])
        return alternatives[i]
      else:
        if n.val == AutoBrancher.OPEN:
          self.trace.append(nodes[i])
          self.this_branch.append(alternatives[i])
          return alternatives[i]

  def __iter__(self):
    cnt = 0
    
    while self.root.val==AutoBrancher.OPEN:
      self.this_branch = []
      cnt+=1
      yield self
      # Indicate that current leaf is done
      self.current.val = AutoBrancher.DONE
      # Close leaves when subleaves are done
      for n in reversed(self.trace[:-1]):
        finished = True
        for e in n.nodes:
          finished = finished and e and e.val==AutoBrancher.DONE
        if finished:
          n.val = AutoBrancher.DONE
      # Reset trace
      self.trace = [self.root]
      print("Evaluated branch",self.this_branch)
    print("Evaluated",cnt,"branches")

def vvcat(arg):
    if len(arg)==0:
        return cs.MX(0,1)
    else:
        return cs.vvcat(arg)

def vcat(arg):
    if len(arg)==0:
        return cs.MX(0,1)
    else:
        return cs.vcat(arg)

def prepare_build_dir(build_dir_abs):
    import os
    import shutil
    os.makedirs(build_dir_abs,exist_ok=True)
    # Clean directory (but do not delete it,
    # since this confuses open shells in Linux (e.g. bash, Matlab)
    for filename in os.listdir(build_dir_abs):
      file_path = os.path.join(build_dir_abs, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
        elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
      except:
        pass
    


class ConstraintInspector:
    def __init__(self, method, stage):
        self.opti = Opti()

        self.X = self.opti.variable(*stage.x.shape)
        self.U = self.opti.variable(*stage.u.shape)
        self.V = self.opti.variable(*stage.v.shape)
        self.P = self.opti.parameter(*stage.p.shape)
        self.t = self.opti.variable()
        self.T = self.opti.variable()

        offsets = list(stage._offsets.keys())

        self.offsets = []
        for e in offsets:
            self.offsets.append(self.opti.variable(*e.shape))

        self.raw = [stage.x,stage.u,stage.p,stage.t, method.v]+offsets
        self.optivar = [self.X, self.U, self.P, self.t, self.V]+self.offsets

        if method.free_time:
            self.raw += [stage.T]
            self.optivar += [self.T]

        self.method = method
    
    def finalize(self):
        if hasattr(self.method,'signals'):
            # Need to a add signal derivative symbols
            extra_vars = []
            extra_pars = []
            for k,v in self.method.signals.items():
                if v.derivative_of is None: continue
                if v.parametric:
                    extra_pars.append(v.symbol)
                else:
                    extra_vars.append(v.symbol)
            raw_var = vvcat(extra_vars)
            raw_par = vvcat(extra_pars)
            self.raw += [raw_var,raw_par]
            self.optivar += [self.opti.variable(*raw_var.shape),self.opti.parameter(*raw_par.shape)]
   
        self.opti_advanced = self.opti.advanced

    def canon(self,expr):
        c = substitute([expr],self.raw,self.optivar)[0]
        mc = self.opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
        return substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw), mc


def linear_coeffs(expr, *args):
    """ Multi-argument extesion to CasADi linear_coeff"""
    J,c = linear_coeff(expr, vcat(args))
    cs = np.cumsum([0]+[e.numel() for e in args])
    return tuple([J[:,cs[i]:cs[i+1]] for i in range(len(args))])+(c,)

ca_classes = [cs.SX,cs.MX,cs.Function,cs.Sparsity,cs.DM,cs.Opti]

@contextmanager
def rockit_pickle_context():
    string_serializer = cs.StringSerializer()
    string_serializer.pack("preamble")
    string_serializer.encode()
    def __getstate__(self):
        if isinstance(self,cs.Opti):
            raise Exception("Opti cannot be serialized yet. Consider removing the transcribed problem.")
        string_serializer.pack(self)
        enc = string_serializer.encode()
        return {"s":enc}

    for c in ca_classes:
        setattr(c, '__getstate__', __getstate__)
        
    yield

    for c in ca_classes:
        delattr(c, '__getstate__')

@contextmanager
def rockit_unpickle_context():
    string_serializer = cs.StringSerializer()
    string_serializer.pack("preamble")
    string_deserializer = cs.StringDeserializer(string_serializer.encode())
    string_deserializer.unpack()
    def __setstate__(self,state):
        string_deserializer.decode(state["s"])
        s = string_deserializer.unpack()
        self.this = s.this

    for c in ca_classes:
        setattr(c, '__setstate__', __setstate__)
        
    yield

    for c in ca_classes:
        delattr(c, '__setstate__')
        