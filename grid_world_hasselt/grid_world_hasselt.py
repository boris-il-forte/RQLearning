import numpy as np
import os
from joblib import Parallel, delayed
from mushroom.algorithms.value.td import RQLearning
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter
from mushroom.utils.variance_parameters import VarianceIncreasingParameter, \
    WindowedVarianceIncreasingParameter

from collectParameters import CollectParameters


def experiment(decay_exp, alphaType):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                                        shape=mdp.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)


    # Agent
    shape = mdp.observation_space.size + mdp.action_space.size
    if alphaType == 'Decay':
        alpha = ExponentialDecayParameter(value=1, decay_exp=decay_exp,
                                          shape=shape)
    else:
        alpha = VarianceIncreasingParameter(value=1, shape=shape, tol=100.)
    #beta = VarianceIncreasingParameter(value=1, shape=shape, tol=1.)
    beta = WindowedVarianceIncreasingParameter(value=1, shape=shape, tol=1.,
                                               window=50)
    #delta = VarianceDecreasingParameter(value=0, shape=shape)
    algorithm_params = dict(learning_rate=alpha, beta=beta, off_policy=True)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = RQLearning(shape, pi, mdp.gamma, agent_params)

    # Algorithm
    collect_max_Q = CollectMaxQ(agent.Q, np.array([mdp._start]))
    collect_dataset = CollectDataset()
    collect_lr = CollectParameters(beta)
    callbacks = [collect_max_Q, collect_dataset, collect_lr]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get_values()
    lr = collect_lr.get_values()

    return reward, max_Qs, lr


if __name__ == '__main__':
    n_experiment = 1

    os.mkdir('/tmp/mushroom')

    names = {1: '1', 0.8: '08'}
    exp = [1, 0.8]
    for e in exp:
        out = Parallel(n_jobs=-1)(delayed(
            experiment)(e, 'Decay') for _ in xrange(n_experiment))
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])
        lr = np.array([o[2] for o in out])

        np.save('results/rQDecWin' + names[e] + '.npy', np.convolve(np.mean(r,
                                                                            0),
                                                            np.ones(100) / 100.,
                                                            'valid'))
        np.save('results/maxQDecWin' + names[e] + '.npy', np.mean(max_Qs, 0))
        np.save('results/lrQDecWin' + names[e] + '.npy', np.mean(lr, 0))

    out = Parallel(n_jobs=-1)(delayed(
        experiment)(0, '') for _ in xrange(n_experiment))
    r = np.array([o[0] for o in out])
    max_Qs = np.array([o[1] for o in out])
    lr = np.array([o[2] for o in out])

    np.save('results/rQDecWinAlpha.npy', np.convolve(np.mean(r, 0),
                                                     np.ones(100) / 100.,
                                                     'valid'))
    np.save('results/maxQDecWinAlpha.npy', np.mean(max_Qs, 0))
    np.save('results/lrQDecWinAlpha.npy', np.mean(lr, 0))
