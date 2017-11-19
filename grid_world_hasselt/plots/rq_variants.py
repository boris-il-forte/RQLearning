import numpy as np
from matplotlib import pyplot as plt
#from matplotlib2tikz import save as tikz_save

alg = ['RQ', 'RQ_Win', 'RQ_Delta']
c = ['brown', 'lawngreen', 'green']
s = ['-', '-', '--']
exp = ['1', '08']
step = 20

base_folder = 'results/'
#base_folder = '/tmp/mushroom/grid_world_hasselt/'
for e in exp:
    plt.figure()
    plt.suptitle(e)
    for i, a in enumerate(alg):
        plt.subplot(2, 1, 1)
        plt.plot(np.load(base_folder+ a + '_' + e + '_r.npy')[::step], c[i], linestyle=s[i], linewidth=3)
        plt.subplot(2, 1, 2)
        plt.plot(np.load(base_folder + a + '_' + e + '_maxQ.npy')[::step], c[i], linestyle=s[i], linewidth=3)

    plt.subplot(2, 1, 1)
    plt.plot(np.load(base_folder+'RQ_Alpha_r.npy')[::step], 'salmon', linewidth=3)
    plt.plot(np.load(base_folder+'RQ_Win_Alpha_r.npy')[::step], 'purple', linestyle='--', linewidth=3)
    plt.plot(np.ones(10000 / step) * 0.2, 'k', linewidth=1)
    plt.subplot(2, 1, 2)
    plt.xlabel('# steps')
    plt.plot(np.load(base_folder+'RQ_Alpha_maxQ.npy')[::step], 'salmon', linewidth=3)
    plt.plot(np.load(base_folder+'RQ_Win_Alpha_maxQ.npy')[::step], 'purple', linestyle='--', linewidth=3)
    plt.plot(np.ones(10000 / step) * 0.36, 'k', linewidth=1)    
    
    if e == '1':
        plt.subplot(2, 1, 1)
        plt.ylabel('Reward')
        plt.subplot(2, 1, 2)
        plt.ylabel('maxQ(s,a)')

    if e == '08':
        plt.subplot(2, 1, 1)
        plt.legend(alg + ['RQ_Alpha', 'RQ_Win_Alpha'])
    #tikz_save('QDecs' + e + '.tex', figureheight='5cm', figurewidth='6cm')
plt.show()
