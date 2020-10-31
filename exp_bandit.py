import numpy as np 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.cm as cm
import time
import copy
import os
import pickle

T = 100000
regret = []


def truep_mean(x,n=100):
    y = np.ceil(x*100)/50
    return  y


xx = np.linspace(0,1,100)
yy = truep_mean(xx)


def truep(x,n=100):
    
    r = np.random.uniform(0,1,1)
    y = np.ceil(x*100)/50*10 if r < 0.1 else 0
    return  y


def boost(t, n, tau):
    c = np.sqrt(4.14*(np.log(2*np.exp(1))/np.log(2)))
    return c*tau*np.sqrt(np.log(t+1)/n)


matplotlib.rcParams.update({'font.size': 22})

# +
xx = np.zeros(T)
K = 100
total = np.zeros(K)        #total is an estimate of Z: sum(p/u)
sigma = np.zeros(K)

for k in range(K):
    total[k] = truep((k+np.random.uniform(0,1,1))/K)
    
nn = np.ones(K);
q = total/sum(total);
for tt in range(T):
    k = np.random.choice(range(K), 1, p = q)[0] #choose the arm  ;
    rr = np.random.uniform(0,1,1)
    xx[tt] = k/K + rr/K
    total[k] = total[k] + truep(xx[tt])
    nn[k] = nn[k] + 1
    tau = 0.1*np.ones(K)

    sigma = [boost(tt,nn[j],tau[j]) for j in range(K)]
    q = (total/nn + sigma)/sum(total/nn + sigma)

# -

plt.plot(q)

yy

q

# +
n=100
qq = q
for i in range(1,(n-1)):
	plt.plot([(i-1)/n,i/n],[yy[i-1],yy[i-1]],color="blue",linewidth=4)
	plt.plot([(i-1)/n,i/n],[n*qq[i-1],n*qq[i-1]],color="red",linewidth=4)

i=n-1
plt.plot([(i-1)/n,i/n],[yy[i],yy[i]],color="blue",label="target",linewidth=4)
plt.plot([(i-1)/n,i/n],[n*qq[i],n*qq[i]],color="red",label="proposal",linewidth=4)
plt.xlabel("x",fontsize=20)
plt.ylabel("density",fontsize=20)
plt.legend(loc="upper right",fontsize=20)
plt.title("target and proposal densities",fontsize=20)
#plt.savefig('results_jcgs/density_no_boost.pdf')
# -

yy


