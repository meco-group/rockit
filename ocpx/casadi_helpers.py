from casadi import *


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


def get_meta():
    # Construct meta-data
    import sys
    import os
    frame = sys._getframe(2)
    meta = {"stacktrace": [{"file":os.path.abspath(frame.f_code.co_filename),"line":frame.f_lineno,"name":frame.f_code.co_name} ] }
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
