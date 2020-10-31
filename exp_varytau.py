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





T = 1000000
K = 10
m=0.1
M=10
path = 'results_varytau_new/'
os.mkdir(path)

total = np.zeros(K)        #total is an estimate of Z: sum(p/u)
KL = np.zeros(T)
xx = np.zeros(T)
sigma = np.zeros(K)

Tau, out,regret = [],[],[]
for delta in np.linspace(0.001,8,10):
    def truep(x,delta=delta,m=m,M=M):
        if x < 1/K:
            y = M + delta if x < 1/K/2 else M-delta
        else:
            y = m
        return  y
    tau = (M+delta-m)/2/K
   # tau = ((M+delta-m)/2 - (M/K + m*(K-1)/K))/K
    Tau.append(tau)
    Z = M/K + (K-1)*m/K
    pi_a = [M/K]
    pi_a = pi_a + [m/K]*(K-1)
    pi_a = np.reshape(pi_a,len(pi_a))/Z
    def truep(x,delta=delta,m=m,M=M):
        if x < 1/K:
            y = M + delta if x < 1/K/2 else M-delta
        else:
            y = m
        return  y
    def boost(t, n, tau=tau):
        c = np.sqrt(4.14*(np.log(2*np.exp(1))/np.log(2)))
        return c*tau*np.sqrt(np.log(t+1)/n)
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
        sigma = [boost(tt,nn[j]) for j in range(K)]
        q = (total/nn + sigma)/sum(total/nn + sigma)
        KL[tt] = sum(pi_a * np.log(pi_a/q))
    regret.append(KL)     
    out.append(KL[-1])


# ###############
# ############plot

with open(path + 'results.pck', 'wb') as f:
    pickle.dump({'Tau': Tau, 'regret': regret, 'out': out}, f)


#################################################################################
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
path = './'


data = pickle.load( open( "results.pck", "rb" ) )
Tau = data['Tau']
regret = data['regret']
out = data['out']



matplotlib.rcParams.update({'font.size': 30})



fig, ax = plt.subplots(figsize = (8,6))
ax.plot(Tau,out)
#ax.set_title("regret")
ax.set_xlabel(r'$\tau$')
ax.set_ylabel("Regret")
#ax.set_xscale("log")
#ax.set_yscale("log")
#plt.yticks([])
fig.tight_layout()
#plt.savefig(path + 'vary_tau.png')
#plt.close()
plt.savefig('results_jcgs/regret_tau0.pdf')




fig, ax = plt.subplots(figsize = (8,6))
yy = [truep(x) for x in xxx]
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
plt.savefig(path + 'density.png')
plt.close()

    

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(np.sqrt(range(len(KL))), np.cumsum(KL[0:]))
ax.set_title("Cumulative Regret")
ax.set_xlabel("Iteration")
ax.set_ylabel("Regret")
#ax.set_xscale("log")
#ax.set_yscale("log")
fig.tight_layout()
plt.savefig(path + 'regret.png')
plt.close()


