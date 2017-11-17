import numpy as np
from joblib import Parallel, delayed

from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.variance_parameters import WindowedVarianceIncreasingParameter
from mushroom.utils.parameters import ExponentialDecayParameter
from mushroom.algorithms.value.td import SARSA, RQLearning
from mushroom.utils.folder import mk_dir_recursive


def experiment1(decay_exp):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                                        size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    alpha = ExponentialDecayParameter(value=1, decay_exp=decay_exp, size=mdp.info.size)
    beta = WindowedVarianceIncreasingParameter(value=1, size=mdp.info.size, tol=1., window=50)
    algorithm_params = dict(learning_rate=alpha, beta=beta, off_policy=False)
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

def experiment2(decay_exp):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

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
    agent = SARSA(pi, mdp.info, agent_params)

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
    for e in exp:
        print 'Off-policy RQ_Win'
        out = Parallel(n_jobs=-1)(delayed(
            experiment1)(e) for _ in xrange(n_experiment))
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])

        np.save(base_folder + 'RQ_Win_onpolicy_'+ names[e] +'_r.npy', np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
        np.save(base_folder + 'RQ_Win_onpolicy_'+ names[e] +'_maxQ.npy', np.mean(max_Qs, 0))

        print 'SARSA'
        out = Parallel(n_jobs=-1)(delayed(
            experiment2)(e) for _ in xrange(n_experiment))
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])

        np.save(base_folder + 'SARSA_' + names[e] + '_r.npy', np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid'))
        np.save(base_folder + 'SARSA_' + names[e] + '_maxQ.npy', np.mean(max_Qs, 0))