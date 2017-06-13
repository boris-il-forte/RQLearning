class DeltaParameter(object):
    def __init__(self, value, exponential=False, min_value=None, shape=(1,)):
        self._initial_value = value
        self._exponential = exponential
        self._min_value = min_value
        self._Q2 = np.zeros(shape)
        self._n_updates = np.zeros(shape)
        self._weights_var = np.zeros(shape)
        self._sigmas = np.ones(shape)*1e10

    def __call__(self, idx, val, update=True):
        if isinstance(idx, list):
            assert len(idx) == 2

            idx = np.concatenate((idx[0].astype(np.int),
                                  idx[1].astype(np.int)),
                                 axis=1).ravel()
        else:
            idx = idx.astype(np.int)
        assert idx.ndim == 1

        idx = tuple(idx) if idx.size == self._n_updates.ndim else 0

        return self._compute(idx, val)

    def _compute(self, idx, alpha, Q, target):

        self._Q2[idx]= (1.0 - alpha) * self._Q2[idx] + alpha * target**2;

        if (self._n_updates[idx] > 1.0):
            self._weights_var[idx] = (1.0 - alpha)**2 * self._weights_var[idx] + alpha**2.0
            n = 1.0 / self._weights_var[idx]
            diff = self._Q2(idx) - Q[idx]^2.0
            if diff < 0.0:
                diff = 0.0
            self._sigmas[idx] = np.sqrt(diff / n);


        new_value = 1.0 - np.exp(-self._sigmas[idx]) if self._exponential else 1.0 / (self.sigmas[idx] + 1.0)
        if self._min_value is None or new_value >= self._min_value:
            return new_value
        else:
            return self._min_value