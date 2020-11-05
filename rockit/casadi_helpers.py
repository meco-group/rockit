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

from casadi import *
from collections import defaultdict, OrderedDict

def get_ranges_dict(list_expr):
    ret = {}
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
        