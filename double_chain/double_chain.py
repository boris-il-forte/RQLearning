import numpy as np
from joblib import Parallel, delayed
from mushroom.algorithms.value.td import RQLearning, DoubleQLearning, SpeedyQLearning, WeightedQLearning, QLearning
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectQ, CollectParameters
from mushroom.utils.parameters import Parameter, ExponentialDecayParameter
from mushroom.utils.variance_parameters import VarianceIncreasingParameter, WindowedVarianceIncreasingParameter
from mushroom.utils.folder import mk_dir_recursive


def experiment1(decay_exp, beta_type):
    np.random.seed()

    # MDP
    p = np.load('p.npy')
    rew = np.load('rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    alpha = ExponentialDecayParameter(value=1, decay_exp=decay_exp, size=mdp.info.size)

    if beta_type == 'Win':
        beta = WindowedVarianceIncreasingParameter(value=1, size=mdp.info.size, tol=10., window=50)
    else:
        beta = VarianceIncreasingParameter(value=1, size=mdp.info.size, tol=10.)

    algorithm_params = dict(learning_rate=alpha, beta=beta, off_policy=True)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = RQLearning(pi, mdp.info, agent_params)

    # Algorithm
    collect_q = CollectQ(agent.Q)
    collect_lr_1 = CollectParameters(beta, np.array([0]))
    collect_lr_5 = CollectParameters(beta, np.array([4]))
    callbacks = [collect_q, collect_lr_1, collect_lr_5]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=20000, n_steps_per_fit=1, quiet=True)

    Qs = collect_q.get_values()
    lr_1 = collect_lr_1.get_values()
    lr_5 = collect_lr_5.get_values()

    return Qs, lr_1, lr_5




def experiment2(algorithm_class, decay_exp):
    np.random.seed()

    # MDP
    p = np.load('p.npy')
    rew = np.load('rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1, decay_exp=decay_exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(pi, mdp.info, agent_params)

    # Algorithm
    collect_Q = CollectQ(agent.Q)
    callbacks = [collect_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=20000, n_steps_per_fit=1, quiet=True)
    Qs = collect_Q.get_values()

    return Qs

if __name__ == '__main__':
    n_experiment = 500

    base_folder = '/tmp/mushroom/double_chain/'
    mk_dir_recursive(base_folder)

    names = {1: '1', .51: '51', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    exps = [1, .51]
    algs = [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning]
    beta_types=['', 'Win']

    for e in exps:
        for a in algs:
            print names[a] + '_' + names[e]
            out = Parallel(n_jobs=-1)(
                delayed(experiment2)(a, e) for _ in xrange(n_experiment))
            Qs = np.array(out)

            Qs = np.mean(Qs, 0)

            np.save(base_folder + names[a] + '_' + names[e] + '_Q.npy', Qs)
            del Qs

        for t in beta_types:
            alg_name = 'RQ'
            if t == 'Win':
                alg_name += '_Win'

            print alg_name + '_' + names[e]
            out = Parallel(n_jobs=-1)(delayed(
                experiment1)(e, t) for _ in xrange(n_experiment))
            Qs = np.array([o[0] for o in out])
            lr_1 = np.array([o[1] for o in out])
            lr_5 = np.array([o[2] for o in out])

            Qs = np.mean(Qs, 0)
            lr_1 = np.mean(lr_1, 0)
            lr_5 = np.mean(lr_5, 0)

            np.save(base_folder + alg_name + '_' + names[e] + '_Q.npy', Qs)
            np.save(base_folder + alg_name + '_' + names[e] + '_lr_1.npy', lr_1)
            np.save(base_folder + alg_name + '_' + names[e] + '_lr_5.npy', lr_5)

            del Qs
            del lr_1
            del lr_5
