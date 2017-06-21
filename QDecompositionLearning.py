import numpy as np

from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import max_QA, parse_dataset, state_action,\
    state_action_idx


class QDecompositionLearning(Agent):
    """
    Implements functions to run QDec algorithms.
    """
    def __init__(self, approximator, policy, states, actions, **params):

        self.offpolicy = params['algorithm_params'].pop('offpolicy')
        self.alpha = params['algorithm_params'].pop('learning_rate')

        self.delta = params['algorithm_params'].get('delta')

        if self.delta is not None:
            del params['algorithm_params']['delta']
        else:
            self.beta = params['algorithm_params'].pop('beta')

        self.r_tilde = np.zeros(shape=states+actions)
        self.q_tilde = np.zeros(shape=states+actions)

        super(QDecompositionLearning, self).__init__(approximator, policy, **params)

    def fit(self, dataset, n_fit_iterations=1):
        """
        Single fit step.
        """
        assert n_fit_iterations == 1
        state, action, reward, next_state, absorbing, _ = parse_dataset(
            [dataset[-1]])

        sa_idx = state_action_idx(state, action)
        sa = state_action(state, action)
        sa1 = state_action_idx(state, action)


        # Reward update
        alpha = self.alpha(sa_idx, target=reward)
        self.r_tilde[sa1] += alpha * (reward - self.r_tilde[sa1])

        # Q update
        if not absorbing:
            q_next = self._next_q(next_state)

            if self.delta is not None:
                beta = alpha * self.delta(sa_idx, target=q_next)
            else:
                beta = self.beta(sa_idx, target=q_next)

            self.q_tilde[sa1] += beta * (q_next - self.q_tilde[sa1])

        # Update policy
        q = self.r_tilde[sa1] + self.mdp_info['gamma']*self.q_tilde[sa1]
        self.approximator.fit(sa, np.array([q]), **self.params['fit_params'])

    def __str__(self):
        return self.__name__

    def _next_q(self,next_state):
        if self.offpolicy:
            max_q,_ = max_QA(next_state, False,
                            self.approximator,
                            self.mdp_info['action_space'].values)
            return max_q
        else:
            self._next_action = self.draw_action(next_state)
            sa_n = [next_state, self._next_action]

            return self.approximator.predict(sa_n)


