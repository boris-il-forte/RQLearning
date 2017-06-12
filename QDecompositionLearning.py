import numpy as np

from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import max_QA, parse_dataset


class QDecompositionLearning(Agent):
    """
    Implements functions to run QDec algorithms.
    """
    def __init__(self, approximator, policy, actions, states, **params):

        self.alpha = params['algorithm_params'].pop('learning_rate')
        self.delta = params['algorithm_params'].pop('delta')
        self.offpolicy = params['algorithm_params'].pop('offpolicy')

        self.r_tilde = np.zeros(shape=actions+states)
        self.q_tilde = np.zeros(shape=actions+states)

        super(QDecompositionLearning, self).__init__(approximator, policy, **params)


    def fit(self, dataset, n_fit_iterations=1):
        """
        Single fit step.
        """
        assert n_fit_iterations == 1
        state, action, reward, next_state, absorbing, _ = parse_dataset(
            [dataset[-1]])

        sa = [state, action]
        sa1 = np.concatenate((state, action), axis=1)

        alpha = self.alpha(sa)

        # Reward update
        r_current = self.r_tilde[sa1]
        self.r_tilde[sa1] = r_current + alpha*(reward - r_current)

        # Q update
        delta = self.delta(sa)
        qtilde_current = self.q_tilde[sa1]
        q_next = self._next_q(next_state) if not absorbing else 0.
        self.q_tilde[sa1] = qtilde_current + alpha*delta * (q_next - qtilde_current)

        # Update policy
        q = self.r_tilde + self.mdp_info['gamma']*self.q_tilde
        self.approximator.fit(sa,q,**self.params['fit_params'])

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


