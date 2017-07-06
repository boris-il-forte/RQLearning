import numpy as np
from joblib import Parallel, delayed

from QDecompositionLearning import QDecompositionLearning
from VarianceParameter import VarianceDecreasingParameter,\
    VarianceIncreasingParameter
from WindowedVarianceParameter import WindowedVarianceIncreasingParameter
from collectParameters import CollectParameters

from PyPi.approximators import Ensemble, Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.callbacks import CollectMaxQ
from PyPi.utils.dataset import compute_J, parse_dataset
from PyPi.utils.parameters import DecayParameter


def experiment():
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = DecayParameter(value=1, decay_exp=.5,
                             shape=mdp.observation_space.shape)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.shape + mdp.action_space.shape
    approximator_params = dict(shape=shape)
    approximator = Regressor(Tabular, **approximator_params)

    # Agent
    alpha = DecayParameter(value=1, decay_exp=.8, shape=shape)
    #alpha = VarianceIncreasingParameter(value=1, shape=shape, tol=100.)
    #beta = VarianceIncreasingParameter(value=1, shape=shape, tol=1.)
    beta = WindowedVarianceIncreasingParameter(value=1, shape=shape, tol=1., window=50)
    #delta = VarianceDecreasingParameter(value=0, shape=shape)
    algorithm_params = dict(learning_rate=alpha, beta=beta, offpolicy=True)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QDecompositionLearning(approximator, pi,
                                   mdp.observation_space.shape,
                                   mdp.action_space.shape, **agent_params)

    # Algorithm
    collect_max_Q = CollectMaxQ(approximator, np.array([mdp._start]),
                                mdp.action_space.values)
    collect_lr = CollectParameters(beta)
    callbacks = [collect_max_Q, collect_lr]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(core.get_dataset())
    max_Qs = collect_max_Q.get_values()
    lr = collect_lr.get_values()

    return reward, max_Qs, lr

if __name__ == '__main__':
    n_experiment = 10000

    logger.Logger(3)

    out = Parallel(n_jobs=-1)(delayed(
        experiment)() for _ in xrange(n_experiment))
    r = np.array([o[0] for o in out])
    max_Qs = np.array([o[1] for o in out])
    lr = np.array([o[2] for o in out])

    np.save('rQDec.npy',
            np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
    np.save('maxQDec.npy', np.mean(max_Qs, 0))
    np.save('lrQDec.npy', np.mean(lr, 0))
