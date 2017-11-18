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
    WindowedVarianceIncreasingParameter, VarianceDecreasingParameter
from mushroom.utils.folder import mk_dir_recursive


def experiment(decay_exp, alphaType, useDelta, windowed):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                                        size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)


    # Agent
    if alphaType == 'Decay':
        alpha = ExponentialDecayParameter(value=1, decay_exp=decay_exp,
                                          size=mdp.info.size)
    else:
        alpha = VarianceIncreasingParameter(value=1, size=mdp.info.size, tol=100.)

    if useDelta:
        delta = VarianceDecreasingParameter(value=0, size=mdp.info.size)
        algorithm_params = dict(learning_rate=alpha, delta=delta, off_policy=True)
    else:
        if windowed:
            beta = WindowedVarianceIncreasingParameter(value=1, size=mdp.info.size, tol=1., window=50)
        else:
            beta = VarianceIncreasingParameter(value=1, size=mdp.info.size, tol=1.)

        algorithm_params = dict(learning_rate=alpha, beta=beta, off_policy=True)

    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = RQLearning(pi, mdp.info, agent_params)

    # Algorithm
    collect_max_Q = CollectMaxQ(agent.Q, mdp.convert_to_int(mdp._start, mdp._width))
    collect_dataset = CollectDataset()
    callbacks = [collect_max_Q, collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1, quiet=True)

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get_values()

    return reward, max_Qs


if __name__ == '__main__':
    n_experiment = 10000

    base_folder = '/tmp/mushroom/grid_world_hasselt/'
    mk_dir_recursive(base_folder)

    names = {1: '1', 0.8: '08'}
    exp = [1, 0.8]

    for windowed in [True, False]:
        windowed_name = 'Win_' if windowed else ''

        # RQ with alpha decay
        for e in exp:
            print 'RQ_' + windowed_name + names[e]
            out = Parallel(n_jobs=-1)(delayed(
                experiment)(e, 'Decay', False, windowed) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            np.save(base_folder+'RQ_' + windowed_name + names[e] + '_r.npy', np.convolve(np.mean(r, 0),
                                                                np.ones(100) / 100.,'valid'))
            np.save(base_folder + 'RQ_' + windowed_name + names[e] + '_maxQ.npy', np.mean(max_Qs, 0))

        #RQ with alpha variance dependent
        print 'RQ_' + windowed_name + 'Alpha'
        out = Parallel(n_jobs=-1)(delayed(
            experiment)(0, '', False, windowed) for _ in xrange(n_experiment))
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])

        np.save(base_folder + 'RQ_' + windowed_name + 'Alpha_r.npy', np.convolve(np.mean(r, 0),
                                                         np.ones(100) / 100.,
                                                         'valid'))
        np.save(base_folder + 'RQ_' + windowed_name + 'Alpha_maxQ.npy', np.mean(max_Qs, 0))

    # RQ with delta
    for e in exp:
        print 'RQ_Delta_' + names[e]
        out = Parallel(n_jobs=-1)(delayed(
            experiment)(e, 'Decay', True, False) for _ in xrange(n_experiment))
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])

        np.save(base_folder + 'RQ_Delta_' + names[e] + '_r.npy', np.convolve(np.mean(r, 0),
                                                                np.ones(100) / 100., 'valid'))
        np.save(base_folder + 'RQ_Delta_' + names[e] + '_maxQ.npy', np.mean(max_Qs, 0))
