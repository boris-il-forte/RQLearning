import numpy as np
from joblib import Parallel, delayed

from QDecompositionLearning import QDecompositionLearning
from VarianceParameter import VarianceDecreasingParameter,\
    VarianceIncreasingParameter
from collectParameters import CollectParameters
from PyPi.algorithms.td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning
from PyPi.approximators import Ensemble, Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils.callbacks import CollectMaxQ, CollectQ
from PyPi.utils import logger
from PyPi.utils.dataset import parse_dataset
from PyPi.utils.parameters import Parameter, DecayParameter


def experiment1(decay_exp):
    np.random.seed()

    # MDP
    p = np.load('p.npy')
    rew = np.load('rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.shape + mdp.action_space.shape
    approximator_params = dict(shape=shape)
    approximator = Regressor(Tabular, **approximator_params)

    # Agent
    alpha = DecayParameter(value=1, decay_exp=decay_exp, shape=shape)
    #alpha = Parameter(value=.1)
    #alpha = VarianceIncreasingParameter(value=1, shape=shape, tol=100.)
    beta = VarianceIncreasingParameter(value=1, shape=shape, tol=1.)
    #beta = Parameter(value=1)
    #delta = VarianceDecreasingParameter(value=0, shape=shape)
    algorithm_params = dict(learning_rate=alpha, beta=beta, offpolicy=True)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QDecompositionLearning(approximator, pi,
                                   mdp.observation_space.shape,
                                   mdp.action_space.shape, **agent_params)

    # Algorithm
    collect_Q = CollectQ(approximator)
    collect_lr = CollectParameters(beta)
    callbacks = [collect_Q, collect_lr]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=5000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(core.get_dataset())
    Qs = collect_Q.get_values()
    lr = collect_lr.get_values()

    return Qs, lr


def experiment2(algorithm_class, decay_exp):
    np.random.seed()

    # MDP
    p = np.load('p.npy')
    rew = np.load('rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.shape + mdp.action_space.shape
    approximator_params = dict(shape=shape)
    if algorithm_class is DoubleQLearning:
        approximator = Ensemble(Tabular, 2, **approximator_params)
    else:
        approximator = Regressor(Tabular, **approximator_params)

    # Agent
    learning_rate = DecayParameter(value=1, decay_exp=decay_exp, shape=shape)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(approximator, pi, **agent_params)

    # Algorithm
    collect_Q = CollectQ(approximator)
    callbacks = [collect_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=5000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    Qs = collect_Q.get_values()

    return Qs

if __name__ == '__main__':
    n_experiment = 100

    logger.Logger(3)

    names = {1: '1', .51: '51', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    exps = [1, .51]
    algs = [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning]

    for e in exps:
        for a in algs:
            out = Parallel(n_jobs=-1)(
                delayed(experiment2)(a, e) for _ in xrange(n_experiment))
            Qs = np.array([o[1] for o in out])

            Qs = np.mean(Qs, 0)

            np.save(names[a] + names[e] + '.npy', Qs)

        out = Parallel(n_jobs=-1)(delayed(
            experiment1)(e) for _ in xrange(n_experiment))
        Qs = np.array([o[1] for o in out])
        lr = np.array([o[2] for o in out])

        Qs = np.mean(Qs, 0)
        lr = np.mean(lr, 0)

        np.save('QDec' + names[e] + '.npy', Qs)
        np.save('lrQDec' + names[e] + '.npy', lr)
