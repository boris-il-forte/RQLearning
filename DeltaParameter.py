import numpy as np

class DeltaParameter(object):
    def __init__(self, value, exponential=False, min_value=None, shape=(1,)):
        self._initial_value = value
        self._exponential = exponential
        self._min_value = min_value
        self._n_updates = np.zeros(shape)
        self._weights_var = np.zeros(shape)

    def __call__(self, idx, **kwargs):
        return self._compute(idx, kwargs['Sigma'])

    def _compute(self, idx, Sigma):
        self._n_updates[idx] += 1

        Sigma_estimator = Sigma*self._weights_var[idx]

        if self._n_updates[idx] < 2:
            parameter_value = self._initial_value
        else:
            parameter_value = np.exp(-Sigma_estimator) if self._exponential else 1.0 / (Sigma_estimator + 1.0)

        if self._min_value is not None and parameter_value < self._min_value:
            parameter_value = self._min_value

        self._weights_var[idx] = (1.0 - parameter_value)**2 * self._weights_var[idx] + parameter_value**2.0

        return parameter_value
