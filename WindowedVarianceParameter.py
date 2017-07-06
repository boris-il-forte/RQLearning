import numpy as np

from PyPi.utils.parameters import Parameter
from math import sqrt, log


class WindowedVarianceParameter(Parameter):
    def __init__(self, value, exponential=False, min_value=None, tol=1.0, window=100,
                 shape=(1,)):
        self._exponential = exponential
        self._tol = tol
        self._weights_var = np.zeros(shape)
        self._samples = np.zeros(shape + (window,))
        self._index = np.zeros(shape, dtype=int)
        self._window = window
        self._parameter_value = np.ones(shape)

        super(WindowedVarianceParameter, self).__init__(value, min_value, shape)

    def _compute(self, idx, **kwargs):
        return self._parameter_value[idx]

    def _update(self, idx, **kwargs):
        x = kwargs['target']
        factor = kwargs.get('factor', 1.0)

        # compute parameter value
        n = self._n_updates[idx] - 1

        if n < 2:
            parameter_value = self._initial_value
        else:
            samples = self._samples[idx]

            if n < self._window:
                samples = samples[:int(n)]

            var = np.var(samples)
            var_estimator = var * self._weights_var[idx]
            parameter_value = self._compute_parameter(var_estimator, sigma_process=var, index=idx)

        # update state
        self._samples[idx][self._index[idx]] = x
        self._index[idx] += 1
        if self._index[idx] >= self._window:
            self._index[idx] = 0

        self._weights_var[idx] = (1.0 - factor*parameter_value) ** 2 * self._weights_var[idx] + (factor*parameter_value) ** 2.0
        self._parameter_value[idx] = parameter_value

    def _compute_parameter(self, sigma):
        pass


class WindowedVarianceIncreasingParameter(WindowedVarianceParameter):
    def _compute_parameter(self, sigma, **kwargs):
        return 1 - np.exp(sigma * log(0.5)/self._tol) if self._exponential else sigma / (sigma + self._tol)
