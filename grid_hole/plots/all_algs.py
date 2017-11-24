import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save

alg = ['Q', 'DQ', 'WQ', 'SPQ', 'RQ', 'RQ_Win']
alg_name = ['Q', 'DQ', 'WQ', 'SQ', 'RQ', 'RQ_Win']
c = ['blue', 'red', 'cyan', 'orange', 'lawngreen', 'green']
exp = ['1', '08']
step = 20


#base_folder = '/tmp/mushroom/grid_hole/'
base_folder = 'results/'

l = list()
plt.figure()
for idx, e in enumerate(exp):
    for i, a in enumerate(alg):
        r = np.load(base_folder + a + '_' + e + '_r.npy')
        plt.subplot(2, 2, 1 + idx)
        if a == 'RQ_Win':
            p, = plt.plot(r[::step], c[i], linewidth=3, linestyle='--')
        else:
            p, = plt.plot(r[::step], c[i], linewidth=3)
        if idx == 0:
            l.append(p)
        plt.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        q = np.load(base_folder + a + '_' + e + '_maxQ.npy')
        plt.subplot(2, 2, 3 + idx)
        if a == 'RQ_Win':
            plt.plot(q[::step], c[i], linewidth=3, linestyle='--')
        else:
            plt.plot(q[::step], c[i], linewidth=3)
        plt.xlabel('\Large \# steps')
        plt.xticks([0, 250, 500], [0, 5000, 10000])
    
    if e == '1':
        plt.subplot(2, 2, 1 + idx)
        plt.ylabel('\Large Reward')
        plt.subplot(2, 2, 3 + idx)
        plt.ylabel('\Large maxQ(s,a)')

    plt.subplot(2, 2, 1 + idx)
    plt.plot(np.ones(10000 / step) * 1.25, 'k', linewidth=1)
    plt.subplot(2, 2, 3 + idx)
    plt.plot(np.ones(10000 / step) * 4.782969, 'k', linewidth=1)

plt.figlegend(handles=l, labels=alg_name, loc='lower center', ncol=len(alg), frameon=False)
tikz_save('grid_hole.tex', figureheight='5cm', figurewidth='6cm')
plt.show()

