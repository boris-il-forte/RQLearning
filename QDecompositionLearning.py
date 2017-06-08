import numpy as np
from copy import deepcopy

from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import max_QA, parse_dataset


class QDecompositionLearning(Agent):
    """
    Implements functions to run QDec algorithms.
    """
    def __init__(self, approximator, policy, **params):
        self.alpha = params['algorithm_params'].pop('learning_rate')
        self.delta = params['algorithm_params'].pop('delta')
        self.offpolicy = params['algorithm_params'].pop('offpolicy')

        assert self.approximator.n_models == 3, 'The regressor ensemble must' \
                                                ' have exactly 3 models.'


        super(QDecompositionLearning, self).__init__(approximator, policy, **params)

    def fit(self, dataset, n_fit_iterations=1):
        """
        Single fit step.
        """
        assert n_fit_iterations == 1
        state, action, reward, next_state, absorbing, _ = parse_dataset(
            [dataset[-1]])

        sa = [state, action]
        sa_idx = np.concatenate((
            self.mdp_info['observation_space'].get_idx(state),
            self.mdp_info['action_space'].get_idx(action)),
            axis=1)

        alpha = self.alpha(sa_idx)
        # Reward update
        r_current = self.approximator[0].predict(sa)
        rtilde = r_current + alpha*(reward - r_current)
        self.approximator[0].fit(sa, rtilde, **self.params['fit_params'])


        # Q update
        delta = self.delta(sa_idx)
        qtilde_current = self.approximator[1].predict(sa)
        q_next = self._next_q(next_state) if not absorbing else 0.
        qtilde = qtilde_current + alpha*delta * (q_next - qtilde_current)
        self.approximator[1].fit(sa, qtilde, **self.params['fit_params'])

        q = rtilde +  self.mdp_info['gamma']*qtilde

        self.approximator[2].fit(sa,q,**self.params['fit_params'])
    def __str__(self):
        return self.__name__



    def _next_q(self,next_state):
        if self.offpolicy:
            max_q,_ = max_Q(next_state, False,
                            self.approximator[2],
                            self.mdp_info['action_space'].values)
            return max_q
        else:
            self.next_action = super(QDecompositionLearning,self).draw_action(next_state,self.approximator[2])


    def draw_action(self,state,approximator):
        if self.offpolicy:
            return super(QDecompositionLearning,self).draw_action(state,self.approximator[2])
        else:
            return self.next_action


