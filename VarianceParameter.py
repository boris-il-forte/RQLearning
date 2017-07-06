import numpy as np

from PyPi.utils.parameters import Parameter
from math import sqrt, log


class VarianceParameter(Parameter):
    def __init__(self, value, exponential=False, min_value=None, tol=1.0,
                 shape=(1,)):
        self._exponential = exponential
        self._tol = tol
        self._weights_var = np.zeros(shape)
        self._x = np.zeros(shape)
        self._x2 = np.zeros(shape)
        self._parameter_value = np.ones(shape)

        super(VarianceParameter, self).__init__(value, min_value, shape)

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
            var = n * (self._x2[idx] - self._x[idx] ** 2) / (n - 1.0)
            var_estimator = var * self._weights_var[idx]
            parameter_value = self._compute_parameter(var_estimator, sigma_process=var, index=idx)

        # update state
        self._x[idx] += (x - self._x[idx]) / self._n_updates[idx]
        self._x2[idx] += (x ** 2 - self._x2[idx]) / self._n_updates[idx]
        self._weights_var[idx] = (1.0 - factor*parameter_value) ** 2 * self._weights_var[idx] + (factor*parameter_value) ** 2.0
        self._parameter_value[idx] = parameter_value

    def _compute_parameter(self, sigma):
        pass


class VarianceIncreasingParameter(VarianceParameter):
    def _compute_parameter(self, sigma, **kwargs):
        return 1 - np.exp(sigma * log(0.5)/self._tol) if self._exponential else sigma / (sigma + self._tol)


class VarianceDecreasingParameter(VarianceParameter):
    def _compute_parameter(self, sigma, **kwargs):
        return np.exp(sigma * log(0.5)/self._tol) if self._exponential else 1.0 / (sigma + self._tol)


class VarianceIncreasingParameterAutoTol(VarianceParameter):
    def _compute_parameter(self, sigma, **kwargs):
        sigma_process = kwargs['sigma_process']
        return 1 - np.exp(sigma * log(0.5)/self._tol) if self._exponential else sigma / (sigma + sigma_process + 1e-8)


class VarianceIncreasingParameterAutoTol2(VarianceParameter):

    def __init__(self, value, exponential=False, min_value=None, tol=1.0,
                 shape=(1,)):
        self._curTol = tol
        super(VarianceIncreasingParameterAutoTol2, self).__init__(value, exponential, min_value, tol, shape)

    def _compute_parameter(self, sigma, **kwargs):
        idx = kwargs['index']
        old_value = self._parameter_value[idx]
        new_value = 1 - np.exp(sigma * log(0.5)/self._curTol) if self._exponential else sigma / (sigma + self._curTol)

        if abs(new_value-old_value)/old_value < 0.005:
            self._curTol += 1
        elif new_value > old_value:
            self._curTol -= 1
            self._curTol = max(self._curTol, self._tol)

        #print self._curTol

        return new_value


class VarianceDecreasingParameterAutoTol(VarianceParameter):
    def _compute_parameter(self, sigma, **kwargs):
        sigma_process = kwargs['sigma_process']
        return np.exp(sigma * log(0.5)/self._tol) if self._exponential else 1.0 / (sigma + sigma_process + 1e-8)