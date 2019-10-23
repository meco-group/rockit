from casadi import substitute


class TranscribedPlaceholders:
    def __init__(self):
        self.clear()

    def clear(self):
        self.pool = {}
        self.mark_dirty()

    def mark_dirty(self):
        self.is_dirty = True

    def __call__(self, args):
        if not isinstance(args, list):
            return self.__call__([args])[0]

        ks = list(self.pool.keys())
        vs = [self.pool[k] for k in ks]
        return substitute(args, ks, vs)

