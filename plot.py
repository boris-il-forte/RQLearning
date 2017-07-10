import numpy as np
from matplotlib import pyplot as plt


alg=['Q', 'DQ', 'WQ','SPQ']
par =['0.8', '1.0']

p = '1.0'
plt.figure(0)

for a in alg:
    folder_path="/home/alessandro/Scrivania/q-decomposition/results_gridhole/"
    r = np.load(folder_path +'r_'+ p + a +"_simple_gridhole.npy")
    #r = np.load("/home/alessandro/Scrivania/results complex gridhole/r_"+a+".npy")

    #if (a=='QDec'):
    r = np.mean(r, axis=0)

    r =np.convolve(r, np.ones(100) / 100., 'valid')
    plt.plot(r)
    plt.legend(alg)
    plt.ylabel("Mean reward per step")
    plt.xlabel("#steps")


plt.figure(1)
for a in alg:
    folder_path="/home/alessandro/Scrivania/q-decomposition/results_gridhole/"

    q = np.load(folder_path +'maxQ_'+ p + a +"_simple_gridhole.npy")
    #q = np.load("/home/alessandro/Scrivania/results complex gridhole/maxQ_"+a+".npy")

    if (a=='QDec'):
        q = np.mean(q,0)
    plt.plot(q)
    plt.ylabel("maxQ(s,a)")
    plt.xlabel("#steps")

#alg.append('optimum')
plt.plot(np.ones(1000) * 7.29, '--', color='k')
plt.legend(alg)

beta_types = ['VarianceIncreasing', 'WindowedVarianceIncreasing']


i = 2
folder_path = "/home/alessandro/Scrivania/q-decomposition/results_gridhole/"

parQdec = ['08' , '1']
for p in parQdec:
    plt.figure(i)
    for b in beta_types:
        r = np.load(folder_path +'rQDec_'+ b +"_"+p +".npy")
        #r = np.mean(r, axis=0)

        r = np.convolve(r, np.ones(100) / 100., 'valid')
        plt.plot(r)
        plt.legend(beta_types)
        plt.ylabel("Mean reward per step")
        plt.xlabel("#steps")
        plt.title(str(p))
    i+=1

i +=1
folder_path = "/home/alessandro/Scrivania/q-decomposition/results_gridhole/"

parQdec = ['08' , '1']
for p in parQdec:
    plt.figure(i)
    for b in beta_types:
        q = np.load(folder_path +'maxQDec_'+ b +"_"+p +".npy")
        #q = np.mean(q, axis=0)

        #r = np.convolve(r, np.ones(100) / 100., 'valid')
        plt.plot(q)
        plt.legend(beta_types)
        plt.ylabel("MaxQ(s,a)")
        plt.xlabel("#steps")
        plt.title(str(p))
    i+=1



