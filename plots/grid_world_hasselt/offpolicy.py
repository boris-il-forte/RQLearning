import numpy as np
from matplotlib import pyplot as plt
#from matplotlib2tikz import save as tikz_save

alg = ['SARSA', 'RQ_Win']
c = ['blue', 'green']
exp = ['1', '08']
style = [None, '--']
step = 20

base_folder = '/tmp/mushroom/grid_world_hasselt/'

for e in exp:
    plt.figure()
    plt.title(e)
    for i, a in enumerate(alg):
        plt.subplot(2, 1, 1)
        plt.plot(np.load(base_folder + a + '_' + e + '_r.npy')[::step], c[i], linestyle=style[i], linewidth=3)
        plt.subplot(2, 1, 2)
        plt.plot(np.load(base_folder + a + '_'+ e + '_maxQ.npy')[::step], c[i], linestyle=style[i], linewidth=3)

        plt.subplot(2, 1, 1)
        plt.xlim([0, 495])
        plt.xticks([0, 250, 495], [0, 5000, 10000])
        plt.subplot(2, 1, 2)
        plt.xlim([0, 495])
        plt.xticks([0, 250, 495], [0, 5000, 10000])

    plt.subplot(2, 1, 1)
    plt.plot(np.ones(10000 / step) * 0.2, 'k', linewidth=1)
    plt.subplot(2, 1, 2)
    plt.xlabel('\Large \# steps')
    plt.plot(np.ones(10000 / step) * 0.36, 'k', linewidth=1)    
    
    if e == '1':
        plt.subplot(2, 1, 1)
        plt.ylabel('Reward')
        plt.subplot(2, 1, 2)
        plt.ylabel('maxQ(s,a)')

    if e == '08':
        plt.subplot(2, 1, 1)
        plt.legend(alg)
    #tikz_save('sarsa' + e + '.tex', figureheight='5cm', figurewidth='6cm')
plt.show()
