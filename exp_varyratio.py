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
K = 10
m=0.1
M=10
tau = (M-m)/2/K # ((M-m)/2 - (M/K + m*(K-1)/K))/K
path = 'results_varyratio_new/'
os.mkdir(path)

total = np.zeros(K)        #total is an estimate of Z: sum(p/u)
KL = np.zeros(T)
xx = np.zeros(T)
sigma = np.zeros(K)

ratio,out,regret = [],[],[]
for delta in [0,1/(20*K),1/(10*K),1/(5*K), 1.5/(2*K)]:
    Z = M/K + (K-1)*m/K
    Z_ratio = (M*(1/K-delta) + m*delta)/(delta/(K-1)*M + m*(1/K-delta/(K-1)))
    ratio.append(Z_ratio)
    pi_a = [M*(1/K-delta) + m*delta]
    pi_a = pi_a + [M*delta/(K-1) + (1/K-delta/(K-1))*m]*(K-1)
    pi_a = np.reshape(pi_a,len(pi_a))/Z
    def truep(x,delta=delta,m=m,M=M):
        if x < 1/K-delta:
            y = M
        elif x < 1/K:
            y = m
        else:
            y = M if x*K - int(x*K) < delta/(K-1) else m
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

# +
xx = np.linspace(0,1,1000)

def truep(x,delta=1.5/(2*K),m=0.5,M=1):
    if x < 1/K-delta:
        y = M
    elif x < 1/K:
        y = m
    else:
        y = M if x*K - int(x*K) < delta/(K-1) else m
    return  y

    
ff = [truep(xx[c]) for c in range(1000)]
plt.plot(xx,ff)
# -




# ###############
# ############plot

with open(path + 'results.pck', 'wb') as f:
    pickle.dump({'ratio': ratio, 'regret': regret, 'out': out}, f)


import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
path = './'


data = pickle.load( open( "results.pck", "rb" ) )
ratio = data['ratio']
regret = data['regret']
out = data['out']


matplotlib.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(ratio[::-1][1:], out[::-1][1:])
#ax.set_title("regret")
ax.set_xlabel(r'$Z_{max}/Z_{min}$')
ax.set_ylabel("Regret")
#ax.set_xscale("log")
#ax.set_yscale("log")
#plt.yticks([])
fig.tight_layout()
#plt.savefig(path + 'vary_ratio.png')
#plt.close()
plt.savefig('results_jcgs/regret_ratio.pdf')



fig, ax = plt.subplots(figsize = (8,6))
yy = [truep(x) for x in xxx]
#yy = np.array(yy)
Index = np.matrix.nonzero(q)
ax.set_title('Density',fontsize=20)

for k in range(K):
    ss = k/K
    ax.plot((ss, ss+ 1/K),(q[k]*K,q[k]*K),color='blue', linewidth=2.0,alpha=0.5) if k < K-1 else \
        ax.plot((ss, ss+ 1/K),(q[k]*K,q[k]*K),color='blue', linewidth=2.0,alpha=0.5, label ='Proposal')
    xxx = np.linspace(ss,ss+1/K-1e-5,10)
    yy = np.array([truep(x) for x in xxx])
    ax.plot(xxx,yy/Z,'-.',color='red',alpha=0.5) if k < K-1 else ax.plot(xxx,yy/Z,'o',color='red',alpha=0.5, label ='Target')


ax.legend(loc='best')
plt.savefig(path + 'density.png')
plt.close()

    

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(np.sqrt(range(len(KL))), np.cumsum(KL[0:]))
ax.set_title("Cumulative Regret")
ax.set_xlabel("Iteration")
ax.set_ylabel("Regret")
fig.tight_layout()
#ax.set_xscale("log")
#ax.set_yscale("log")
plt.savefig(path + 'regret.png')
plt.close()


