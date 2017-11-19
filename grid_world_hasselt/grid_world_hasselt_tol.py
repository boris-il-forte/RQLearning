import numpy as np
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter
from mushroom.utils.variance_parameters import VarianceIncreasingParameter, WindowedVarianceIncreasingParameter
from mushroom.algorithms.value.td import RQLearning
from joblib import Parallel, delayed
from mushroom.utils.folder import mk_dir_recursive


def experiment(decay_exp, windowed, tol):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                                        size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    alpha = ExponentialDecayParameter(value=1, decay_exp=decay_exp, size=mdp.info.size)
    if windowed:
        beta = WindowedVarianceIncreasingParameter(value=1, size=mdp.info.size, tol=tol, window=50)
    else:
        beta = VarianceIncreasingParameter(value=1, size=mdp.info.size, tol=tol)
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

    names = {0.8: '08', 0.1: '01', 1: '1', 5: '5', 6: '6', 8: '8', 10: '10'}
    exp = [0.8]

    for e in exp:
        # RQ
        tol = [1, 5, 8, 10]
        for t in tol:
            print 'RQ_' + names[e] + '_tol_' + names[t]
            out = Parallel(n_jobs=-1)(delayed(
                experiment)(e, False, t) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])
            del out

            np.save(base_folder + 'RQ_' + names[e] + '_tol_' + names[t] + '_r.npy', np.convolve(np.mean(r, 0),
                                                                                  np.ones(100) / 100., 'valid'))
            np.save(base_folder + 'RQ_' + names[e] + '_tol_' + names[t] + '_maxQ.npy', np.mean(max_Qs, 0))
            del r
            del max_Qs

        # RQ_Win
        tol = [.1, 1, 5, 10]
        for t in tol:
            print 'RQ_Win_' + names[e] + '_tol_' + names[t]
            out = Parallel(n_jobs=-1)(delayed(
                experiment)(e, True, t) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])
            del out

            np.save(base_folder + 'RQ_Win_' + names[e] + '_tol_' + names[t] + '_r.npy', np.convolve(np.mean(r, 0),
                                                                                  np.ones(100) / 100., 'valid'))
            np.save(base_folder + 'RQ_Win_' + names[e] + '_tol_' + names[t] + '_maxQ.npy', np.mean(max_Qs, 0))
            del r
            del max_Qs
