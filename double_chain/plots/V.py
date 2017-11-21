import numpy as np
from matplotlib import pyplot as plt
#from matplotlib2tikz import save as tikz_save

from mushroom.solvers.dynamic_programming import value_iteration

#base_folder = '/tmp/mushroom/double_chain/'
base_folder = 'results/'

alg = ['Q', 'DQ', 'WQ', 'SPQ', 'RQ', 'RQ_Win']
c = ['blue', 'red', 'cyan', 'orange', 'lawngreen', 'green']
exp = ['1', '51']
step = 40

p = np.load('../p.npy')
r = np.load('../rew.npy')
v = value_iteration(p, r, 0.9, 1e-6)

for e in exp:
    plt.figure()
    for i, a in enumerate(alg):
        plt.plot(np.max(np.load(base_folder + a + '_' + e + '_Q.npy')[:, 0, :], axis=1)[::step], c[i], linewidth=3)

    plt.plot(v[0] * np.ones(20000 / step), 'k', linewidth=1)
    plt.xlabel('# steps')
    
    if e == '1':
        plt.ylabel('maxQ(1,a)')

    if e == '51':
        plt.legend(alg)
    #tikz_save('v1-' + e + '.tex', figureheight='5cm', figurewidth='6cm')
for e in exp:
    plt.figure()
    for i, a in enumerate(alg):
        plt.plot(np.max(np.load(base_folder + a + '_' + e + '_Q.npy')[:, 4, :], axis=1)[::step], c[i], linewidth=3)

    plt.plot(v[4] * np.ones(20000 / step), 'k', linewidth=1)
    plt.xlabel('# steps')
    
    if e == '1':
        plt.ylabel('maxQ(5,a)')

    if e == '51':
        plt.legend(alg)
    #tikz_save('v5-' + e + '.tex', figureheight='5cm', figurewidth='6cm')
plt.show()
