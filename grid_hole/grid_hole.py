import numpy as np
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectMaxQ, CollectDataset
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter
from mushroom.algorithms.value.td import RQLearning, QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning
from mushroom.utils.variance_parameters import VarianceIncreasingParameter, WindowedVarianceIncreasingParameter
from mushroom.utils.folder import mk_dir_recursive
from joblib import Parallel, delayed


def experiment(decay_exp, betaType):
    np.random.seed()

    # MDP

    grid_map = "simple_gridmap.txt"
    mdp = GridWorldGenerator(grid_map=grid_map)

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                             size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    alpha = ExponentialDecayParameter(value=1, decay_exp=decay_exp, size=mdp.info.size)

    if betaType == 'Win':
        beta = WindowedVarianceIncreasingParameter(value=1, size=mdp.info.size, tol=.1, window=50)
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

def experiment_others(alg, decay_exp):
    np.random.seed()

    # MDP

    grid_map = "simple_gridmap.txt"
    mdp = GridWorldGenerator(grid_map=grid_map)

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                             size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    alpha = ExponentialDecayParameter(value=1, decay_exp=decay_exp, size=mdp.info.size)

    algorithm_params = dict(learning_rate=alpha)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = alg(pi, mdp.info, agent_params)

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

    base_folder = '/tmp/mushroom/grid_hole/'
    mk_dir_recursive(base_folder)

    names = {1: '1', 0.8: '08', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}
    exp = [1, 0.8]
    beta_types = ['', 'Win']
    alg_list = [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning]

    for e in exp:
        # RQ
        for t in beta_types:
            alg_name = 'RQ'
            if t == 'Win':
                    alg_name += '_Win'
            print alg_name + '_' + names[e]
            out = Parallel(n_jobs=-1)(delayed(
                experiment)(e,t) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])
            del out

            np.save(base_folder + alg_name + '_' + names[e] + '_r.npy',
                    np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
            np.save(base_folder + alg_name + '_' + names[e] + '_maxQ.npy', np.mean(max_Qs, 0))
            del r
            del max_Qs

        # Others algs
        for a in alg_list:
            print names[a] + '_' + names[e]
            out = Parallel(n_jobs=-1)(
                delayed(experiment_others)(a, e) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])
            del out

            np.save(base_folder + names[a] + '_' + names[e] + '_r.npy',
                    np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
            np.save(base_folder + names[a] + '_' + names[e] + '_maxQ.npy', np.mean(max_Qs, 0))
            del r
            del max_Qs
