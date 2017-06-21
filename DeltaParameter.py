import numpy as np

from PyPi.utils.parameters import Parameter


class DeltaParameter(Parameter):
    def __init__(self, value, exponential=False, min_value=None,
                 shape=(1,)):
        self._exponential = exponential
        self._weights_var = np.zeros(shape)
        self._x = np.zeros(shape)
        self._x2 = np.zeros(shape)
        self._parameter_value = np.ones(shape)

        super(DeltaParameter, self).__init__(value, min_value, shape)

    def _compute(self, idx, **kwargs):
        return self._parameter_value[idx]

    def _update(self, idx, **kwargs):
        x = kwargs['target']

        # compute parameter value
        var = self._n_updates[idx]*(self._x2[idx]-self._x[idx]**2)/(self._n_updates[idx] - 1.0)
        var_estimator = var*self._weights_var[idx]

        if self._n_updates[idx] < 2:
            parameter_value = self._initial_value
        else:
            parameter_value = np.exp(-var_estimator) if self._exponential else 1.0 / (var_estimator + 1.0)

        # update state
        self._x[idx] += (x - self._x[idx]) / self._n_updates[idx]
        self._x2[idx] += (x ** 2 - self._x2[idx]) / self._n_updates[idx]
        self._weights_var[idx] = (1.0 - parameter_value) ** 2 * self._weights_var[idx] + parameter_value ** 2.0
        self._parameter_value[idx] = parameter_value
