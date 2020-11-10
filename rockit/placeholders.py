from casadi import substitute, depends_on, vvcat, DM
from .casadi_helpers import HashDict
from collections import defaultdict

class TranscribedPlaceholders:
    def __init__(self):
        self.clear()

    def clear(self):
        self.pool = [HashDict() for i in range(2)]
        self.mark_dirty()

    def mark_dirty(self):
        self.is_dirty = True

    def __getitem__(self, i):
        return self.pool[i-1]

    def __call__(self, args, preference=None):

        if not isinstance(args, list):
            return self.__call__([args])[0]

        def select(value):
            if len(value)==1:
                return list(value.values())[0]
            for k,v in value.items():
                if k in preference:
                    return v
            raise Exception("ambiguous")

        ks = list(self[2])
        vs = [select(self[2][e]) for e in ks]

        k = [k for k in self[1].keys() if k not in self[2]]        
        ks += k
        vs += [select(self[1][e]) for e in k]
        kv = vvcat(ks)

        # No use doing placeholder substitution on DM
        if isinstance(vvcat(args), DM):
            return args

        # Fixed-point iteration
        while depends_on(vvcat(args), kv):
            args = substitute(args, ks, vs)

        return args

