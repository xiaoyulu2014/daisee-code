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

path = 'results_varyk_new/'
#os.mkdir(path)

KL = np.zeros(T)
xx = np.zeros(T)


m0=1
M0=3

K_vec,out,regret = [],[],[]
K_vec = np.linspace(5,100,20)
for K in K_vec:
    K = int(K)
    m = m0*K
    M = M0*K
    tau = (M-m)/2/K #((M-m)/2 - (M*0.2 + m*0.8))/K
    def truep(x,m=m,M=M):
        y = M if x < 0.2 else m
        return  y
    def boost(t, n, tau=tau):
        c = np.sqrt(4.14*(np.log(2*np.exp(1))/np.log(2)))
        return c*tau*np.sqrt(np.log(t+1)/n)
    total = np.zeros(K)        #total is an estimate of Z: sum(p/u)
    sigma = np.zeros(K)
    Z = 0.2*M + 0.8*m
    pi_a = [M/K]*int(K/5)
    pi_a = pi_a + [m/K]*(K-int(K/5))
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
        sigma = [boost(tt,nn[j]) for j in range(K)]
        q = (total/nn + sigma)/sum(total/nn + sigma)
        KL[tt] = sum(pi_a * np.log(pi_a/q))
    regret.append(KL)     
    out.append(KL[-1])

        

# ###############
# ############plot

with open(path + 'results.pck', 'wb') as f:
    pickle.dump({'K_vec': K_vec, 'regret': regret, 'out': out}, f)





import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
path = './'


data = pickle.load( open( "results.pck", "rb" ) )
K_vec = data['K_vec']
regret = data['regret']
out = data['out']


matplotlib.rcParams.update({'font.size': 30})




fig, ax = plt.subplots(figsize = (8,6))
ax.plot(K_vec,out)
#ax.set_title("regret")
ax.set_xlabel(r'$K$')
ax.set_ylabel("Regret")
#plt.yticks([])
#ax.set_xscale("log")
#ax.set_yscale("log")
fig.tight_layout()
#plt.savefig(path + 'out.png')
plt.savefig('results_jcgs/regret_k.pdf')
plt.show()



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


