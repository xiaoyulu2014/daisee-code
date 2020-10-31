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



T = 10000
path = 'results_sensitivity/'
#os.mkdir(path)
matplotlib.rcParams.update({'font.size': 22})
out = []
tau_vec = np.linspace(0,0.5,10)


for k in range(10):
    regret,qs = [],[]
    for ii in range(len(tau_vec)):
        tau = tau_vec[ii]
        KL = np.zeros(T)
        xx = np.zeros(T)
        K=2
        def truep(x):
            y = 20 if x < 0.25 else 3 if x < 0.5 else 1 if x <0.99 else 9
            return  y
        def boost(t, n, tau):
            c = np.sqrt(4.14*(np.log(2*np.exp(1))/np.log(2)))
            return c*tau*np.sqrt(np.log(t+1)/n)
        total = np.zeros(K)        #total is an estimate of Z: sum(p/u)
        sigma = np.zeros(K)
        Z = 0.25*(3+20) + (0.99-0.5)*1 + 0.01*9
        pi_a = np.array([23/2*0.5,(0.99-0.5) + 0.01*9])/Z
        for k in range(K):
            total[k] = truep((k+np.random.uniform(0,1,1))/K)
        nn = np.ones(K);
        q = total/sum(total);	
        for tt in range(T):
            k = np.random.choice(range(K), 1, p = q)[0] #choose the arm  ;
            rr = np.random.uniform(0,1,1)
            xx[tt] = k/K + rr/K
            total[k] = total[k] + truep(xx[tt])/K
            nn[k] = nn[k] + 1
            sigma = [boost(tt,nn[j],tau) for j in range(K)]
            q = (total/nn + sigma)/sum(total/nn + sigma)
            KL[tt] = sum(pi_a * np.log(pi_a/q))
        regret.append(KL)     
        qs.append(q)
    out.append(np.reshape(regret,(len(tau_vec),-1)))

tmp = [np.mean(np.reshape([out[j][k,:] for j in range(10)],(10,-1)),0) for k in range(len(tau_vec))]

# ###############
# ############plot

with open(path + 'results.pck', 'wb') as f:
    pickle.dump({'eachtau': eachtau, 'regret': regret, 'tmp':tmp, 'tau_vec':tau_vec}, f)





import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
path = './'


data = pickle.load( open( "results.pck", "rb" ) )
eachtau = data['eachtau']
regret = data['regret']
tmp = data['tmp']
tau_vec = data['tau_vec']



matplotlib.rcParams.update({'font.size': 30})



out = [np.cumsum(tmp[i])[-1] for i in range(len(tau_vec))]
fig, ax = plt.subplots(figsize = (8,6))
ax.plot(tau_vec, out)
#ax.set_title("Cumulative Regret")
ax.set_xlabel(r"$\tau$")
#plt.yticks([])
ax.set_ylabel("Regret")
fig.tight_layout()
plt.savefig(path + 'sensitivity.png')
#plt.close()




fig, ax = plt.subplots(figsize = (8,6))
ax.plot(np.sqrt(range(len(regret[0])))[0:], np.cumsum(regret[0])[0:],label='Same ' +r' $\tau$')
ax.plot(np.sqrt(range(len(regret[0])))[0:], np.cumsum(regret[1])[0:],label=r'$ \tau_a$ for each arm')
#ax.set_title("Cumulative Regret")
ax.set_xlabel(r"$\sqrt{t}$")
#ax.set_ylabel("Regret")
ax.legend(loc='best')
plt.yticks([])
#ax.set_xscale("log")
#ax.set_yscale("log")
fig.tight_layout()
plt.savefig(path + 'regret.png')
plt.close()




