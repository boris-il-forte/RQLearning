import numpy as np
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils import logger
from mushroom.utils.callbacks import CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import DecayParameter
from mushroom.algorithms.value.td import RQLearning
from mushroom.utils.variance_parameters import VarianceIncreasingParameter, WindowedVarianceIncreasingParameter
from joblib import Parallel, delayed

from grid_world_hasselt.collectParameters import CollectParameters


def experiment(decay_exp, alphaType,betaType):
    np.random.seed()

    # MDP

    grid_map = "simple_gridmap.txt"
    mdp = GridWorldGenerator(grid_map=grid_map)

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
    if alphaType == 'Decay':
        alpha = DecayParameter(value=1, decay_exp=decay_exp, shape=shape)
    else:
        alpha = VarianceIncreasingParameter(value=1, shape=shape, tol=100.)
    if betaType == 'WindowedVarianceIncreasing':
        beta = WindowedVarianceIncreasingParameter(value=1, shape=shape, tol=0.5, window=50)
    else:
        beta = VarianceIncreasingParameter(value=1, shape=shape, tol=0.5)

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
    n_experiment = 1#0000

    logger.Logger(3)

    names = {1: '1', 0.8: '08'}
    exp = [1, 0.8]
    beta_types = ['','Win']
    for b in beta_types:
        for e in exp:
            out = Parallel(n_jobs=-1)(delayed(
                experiment)(e, 'Decay',b) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])
            lr = np.array([o[2] for o in out])

            np.save('results_gridhole/rQDec'+ b + names[e] +'.npy', np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
            np.save('results_gridhole/maxQDec'+ b + names[e] +'.npy', np.mean(max_Qs, 0))
            np.save('results_gridhole/lrQDec'+ b + names[e] +'.npy', np.mean(lr, 0))

    """
    out = Parallel(n_jobs=-1)(delayed(
        experiment)(0, '') for _ in xrange(n_experiment))
    r = np.array([o[0] for o in out])
    max_Qs = np.array([o[1] for o in out])
    lr = np.array([o[2] for o in out])

    np.save('rQDecWinAlpha.npy', np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
    np.save('maxQDecWinAlpha.npy', np.mean(max_Qs, 0))
    np.save('lrQDecWinAlpha.npy', np.mean(lr, 0))
    """