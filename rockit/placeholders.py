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

    def _replace(self,args,ks,vs):
        # No use doing placeholder substitution on DM
        if isinstance(vvcat(args), DM):
            return args

        kv = vvcat(ks)

        # Fixed-point iteration
        while depends_on(vvcat(args), kv):
            args = substitute(args, ks, vs)

        return args

    def __call__(self, args, max_phase=2, preference=None, verbose=False):
        if not isinstance(args, list):
            return self([args],max_phase=max_phase,preference=preference,verbose=verbose)[0]

        if preference is None:
            preference = ['normal','normal.normal']

        if not isinstance(args, list):
            return self.__call__([args])[0]

        def select(value):
            if len(value)==1:
                return list(value.values())[0]
            for k,v in value.items():
                if k in preference:
                    return v
            raise Exception("ambiguous")

        if max_phase==1:
            ks = list(self[1])
            vs = [select(self[1][e]) for e in ks]
        elif max_phase==2:
            ks = list(self[2])
            vs = [select(self[2][e]) for e in ks]

            k = [k for k in self[1].keys() if k not in self[2]]        
            ks += k
            vs += [select(self[1][e]) for e in k]
            kv = vvcat(ks)

        if verbose:
            print(self.pool)
            print(ks,vs)
        return self._replace(args, ks, vs)