from copy import deepcopy

class CollectParameters():
    def __init__(self, alpha):
        self._alpha = alpha
        self._a = list()

    def __call__(self, **kwargs):
        self._a.append(deepcopy(self._alpha._parameter_value.table))

    def get_values(self):
        return self._a
