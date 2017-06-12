import numpy as np
from joblib import Parallel, delayed

from QDecompositionLearning import QDecompositionLearning
from PyPi.approximators import Ensemble, Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.dataset import parse_dataset
from PyPi.utils.parameters import Parameter


def experiment(algorithm_class, decay_exp):
    np.random.seed()

    # MDP
    mdp = GridWorld(height=3, width=3, goal=(2,2))

    # Policy
    epsilon = Parameter(value=1, decay=False)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.shape + mdp.action_space.shape
    approximator_params = dict(shape=shape)
    approximator = Regressor(Tabular, **approximator_params)

    # Agent
    alpha = Parameter(value=1, decay=True, decay_exp=decay_exp,
                              shape=shape)
    delta = Parameter(value=1, decay=False)
    algorithm_params = dict(learning_rate=alpha, delta=delta, offpolicy=True)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QDecompositionLearning(approximator, pi, mdp.observation_space.shape, mdp.action_space.shape, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(core.get_dataset())

    return reward

if __name__ == '__main__':
    n_experiment = 1

    logger.Logger(3)

    names = {1: '1', .8: '08', QDecompositionLearning: 'QD'}
    for e in [1, .8]:
        for a in [QDecompositionLearning]:
            r = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in xrange(n_experiment))
            from matplotlib import pyplot as plt
            plt.plot(np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
            plt.show()
            #np.save('r' + names[a] + names[e] + '.npy',
            #        np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
