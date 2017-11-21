import numpy as np
from matplotlib import pyplot as plt
#from matplotlib2tikz import save as tikz_save

#base_folder = '/tmp/mushroom/double_chain/'
base_folder = 'results/'

alg = ['Q', 'RQ_Win']
alg_label = ['Q', 'RQWin']
c = ['blue', 'green']
exp = ['1', '51']
states = [0, 8]
best_a = [1, 2]
step = 40

for e in exp:
    plt.figure()
    for i, a in enumerate(alg):
        for j, s in enumerate(states):
            plt.subplot(2, len(alg), ((2 * i) + (j + 1)))
            plt.ylim([.95, 2.05])
            max_a = np.argmax(np.load(base_folder + a + '_' + e + '_Q.npy'), axis=2)[::step, s]
            plt.plot(max_a + 1, c[i], linewidth=3)
            plt.plot(np.ones(20000 / step) * best_a[j], '--k', linewidth=1)
            if j == 0:
                plt.ylabel(alg_label[i])
                plt.yticks([1, 2], [1, 2])
            else:
                plt.yticks([1, 2])
                plt.tick_params(
                    axis='y',
                    which='both',
                    bottom='off',
                    top='off',
                    labelleft='off')
            plt.xlim([0, 500])
            if i == 1:
                plt.xlabel('\Large \# steps')
                plt.xticks([0, 250, 500], [0, 10000, 20000])
            else:
                plt.tick_params(
                    axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
            if i == 0:
                plt.title('state ' + str(states[j] + 1))

        #tikz_save('max_a-' + e + '.tex', strict=True, figureheight='5cm', figurewidth='6cm')
plt.show()

