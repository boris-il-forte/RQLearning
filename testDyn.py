from DynamicProgramming import value_iteration, policy_iteration
from PyPi.environments import *

mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                            gamma=0.9)

V = value_iteration(mdp.p, mdp.r, 0.9, 0.01)
print 'Value iteration'
print V
V, pi = policy_iteration(mdp.p, mdp.r, 0.9)
print 'Policy iteration'
print V
print pi