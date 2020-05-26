from casadi import substitute
from .casadi_helpers import HashDict

class TranscribedPlaceholders:
    def __init__(self):
        self.clear()

    def clear(self):
        self.pool = HashDict()
        self.mark_dirty()

    def mark_dirty(self):
        self.is_dirty = True

    def __call__(self, args):
        if not isinstance(args, list):
            return self.__call__([args])[0]

        ks = list(self.pool.keys())
        vs = [self.pool[k] for k in ks]
        return substitute(args, ks, vs)

