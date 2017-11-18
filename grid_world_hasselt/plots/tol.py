import numpy as np
from matplotlib import pyplot as plt
#from matplotlib2tikz import save as tikz_save

algs = ['RQ_Win', 'RQ']
tols1 = ['01', '1', '5', '10']
tols2 = ['1', '5', '8', '10']
exp_legend1 = ['RQWin01', 'RQWin1', 'RQWin5', 'RQWin10']
exp_legend2 = ['RQ1', 'RQ5', 'RQ8', 'RQ10']

ls = ['-', '--', '-.', ':']
colors = ['green', 'lawngreen']

tols = [tols1, tols2]
exp_legend = [exp_legend1, exp_legend2]
step = 20
#step = 1

base_folder = '/tmp/mushroom/grid_world_hasselt/'
#base_folder = 'results/'

for k, a in enumerate(algs):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlim([0, 495])
    plt.xticks([0, 250, 495], [0, 5000, 10000])
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    plt.ylabel('\Large Reward')
    for i, tol in enumerate(tols[k]):
        plt.plot(np.load(base_folder + a + '_08_tol_' + tol + '_r.npy')[::step], colors[k], linestyle=ls[i], linewidth=3)
    plt.plot(np.ones(10000 / step) * 0.2, 'k', linewidth=1)
    plt.legend(exp_legend[k])

    plt.subplot(2, 1, 2)
    plt.xlim([0, 495])
    plt.xticks([0, 250, 495], [0, 5000, 10000])
    plt.xlabel('\Large \# steps')
    plt.ylabel('\Large maxQ(s,a)')

    for i, tol in enumerate(tols[k]):
        plt.plot(np.load(base_folder + a + '_08_tol_' + tol + '_maxQ.npy')[::step], colors[k], linestyle=ls[i],
                 linewidth=3)
    plt.plot(np.ones(10000 / step) * 0.36, 'k', linewidth=1)
    #tikz_save(a + 'Hasselt.tex', figureheight='5cm', figurewidth='6cm')

plt.show()

