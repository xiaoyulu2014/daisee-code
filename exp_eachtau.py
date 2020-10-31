import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.cm as cm
import time
import copy
import os
import pickle



T = 10000
path = 'results_eachtau/'
#os.mkdir(path)
regret = []

matplotlib.rcParams.update({'font.size': 22})



for eachtau in [False,True]:
    KL = np.zeros(T)
    xx = np.zeros(T)
    m=1
    M=2
    K=5
    tau0 = (M-m)/2/K
    def truep(x,m=m,M=M):
        y = M if x < 0.1 else m
        return  y
    def boost(t, n, tau):
        c = np.sqrt(4.14*(np.log(2*np.exp(1))/np.log(2)))
        return c*tau*np.sqrt(np.log(t+1)/n)
    total = np.zeros(K)        #total is an estimate of Z: sum(p/u)
    sigma = np.zeros(K)
    Z = 0.1*M + 0.9*m
    pi_a = [M*0.1 + m*0.1]
    pi_a = pi_a + [m/K]*(K-1)
    pi_a = np.reshape(pi_a,len(pi_a))/Z
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
        tau = 1e-1 * np.ones(K)#np.zeros(K)
        if eachtau:
            tau[0] = tau0
        else:
            tau = tau0*np.ones(K)
        sigma = [boost(tt,nn[j],tau[j]) for j in range(K)]
        q = (total/nn + sigma)/sum(total/nn + sigma)
        KL[tt] = sum(pi_a * np.log(pi_a/q))
    fig, ax = plt.subplots(figsize = (8,6))
    #yy = [truep(x) for x in xxx]
    #yy = np.array(yy)
    Index = np.matrix.nonzero(q)
    ax.set_title('Density',fontsize=20)
    for k in range(K):
        ss = k/K
        ax.plot((ss, ss+ 1/K),(q[k]*K,q[k]*K),color='blue', linewidth=2.0,alpha=0.5) if k < K-1 else \
            ax.plot((ss, ss+ 1/K),(q[k]*K,q[k]*K),color='blue', linewidth=2.0,alpha=0.5, label ='Proposal')
        xxx = np.linspace(ss,ss+1/K,10)
        yy = np.array([truep(x) for x in xxx])
        ax.plot(xxx,yy/Z,'o',color='red',alpha=0.5) if k < K-1 else ax.plot(xxx,yy/Z,'o',color='red',alpha=0.5, label ='Target')
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(path + 'density_%d.png' %eachtau)
    plt.close()
    regret.append(KL)     




# ###############
# ############plot

with open(path + 'results.pck', 'wb') as f:
    pickle.dump({'eachtau': eachtau, 'regret': regret}, f)





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



matplotlib.rcParams.update({'font.size': 22})

matplotlib.rcParams.update({'font.size': 30})


fig, ax = plt.subplots(figsize = (8,6))
ax.plot(np.sqrt(range(len(regret[0])))[0:], np.cumsum(regret[0])[0:],label='Same ' +r' $\tau$')
ax.plot(np.sqrt(range(len(regret[0])))[0:], np.cumsum(regret[1])[0:],label=r'$ \tau_a$ for each arm')
#ax.set_title("Cumulative Regret")
ax.set_xlabel(r"$\sqrt{t}$")
ax.set_ylabel("Regret")
ax.legend(loc='best')
#plt.yticks([])
#ax.set_xscale("log")
#ax.set_yscale("log")
fig.tight_layout()
#plt.savefig(path + 'regret.png')
plt.savefig('results_jcgs/regret_tau.pdf')
plt.show()



