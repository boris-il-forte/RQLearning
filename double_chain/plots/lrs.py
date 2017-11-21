import numpy as np
from matplotlib import pyplot as plt
#from matplotlib2tikz import save as tikz_save

alg = ['RQ', 'RQ_Win']
alg_name = ['RQ - a=1', 'RQ - a=2', 'RQ_Win - a=1', 'RQ_Win - a=2']
c = ['lawngreen', 'green']
exp = ['1', '51']
step = 40

#base_folder = '/tmp/mushroom/double_chain/'
base_folder = 'results/'

for e in exp:
    plt.figure()
    for i, a in enumerate(alg):
        lr_1 = np.load(base_folder + a + '_' + e + '_lr_1.npy')
        plt.plot(lr_1[::step, 0], c[i], linewidth=3)
        plt.plot(lr_1[::step, 1], c[i], linestyle='--', linewidth=3)

    plt.plot(np.zeros(20000 / step), 'k', linewidth=1)
    plt.xlabel('# steps')
    
    if e == '1':
        plt.ylabel('maxQ(1,a)')

    if e == '51':
        plt.legend(alg_name)
    #tikz_save('lrs1-' + e + '.tex', figureheight='5cm', figurewidth='6cm')
for e in exp:
    plt.figure()
    for i, a in enumerate(alg):
        lr_5 = np.load(base_folder + a + '_' + e + '_lr_5.npy')
        plt.plot(lr_5[::step, 0], c[i], linewidth=3)
        plt.plot(lr_5[::step, 1], c[i], linestyle='--', linewidth=3)

    plt.plot(np.zeros(20000 / step), 'k', linewidth=1)
    plt.xlabel('# steps')
    
    if e == '1':
        plt.ylabel('maxQ(5,a)')

    if e == '51':
        plt.legend(alg_name)

    #tikz_save('lrs5-' + e + '.tex', figureheight='5cm', figurewidth='6cm')
plt.show()
