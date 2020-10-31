import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.cm as cm
import time
import copy



T = 100000
a = 0.5
def truep(x):
    return  (x>=0.25)*np.exp(-10*(1-x)) + (x<0.25)*a;

Z = 0.1 - 0.1*np.exp(-7.5) + 0.25*a;

 

explorefactor = 1e-3
threshold = 10
M = 1  
K = 20;

L_max = 100000
total = np.zeros(K)        #total is an estimate of Z: sum(p/u)
KL = np.zeros(T)
xx = np.zeros(T)
sigma = np.zeros(K)


for k in range(K):
    total[k] = truep((k+np.random.uniform(0,1,1)-1)/K)/K;



nn = np.ones(K);
q = total/sum(total);	


def func_pi_a(ss,k):
    if ss+k < 0.25:
        out = a/Z*k
    elif ss>0.25:
        out = 0.1/Z*(truep(ss+k)-truep(ss))
    else:
        out = a/Z*(0.25-ss) + 0.1/Z*(truep(ss+k)-truep(0.25));
    return out

def func_KL(ss,k,q,pi_a):
    if ss+k < 0.25:
        out = (np.log(a/Z)-np.log(1/k)-np.log(q))*pi_a
    elif ss>0.25:
        out = truep(ss+k)*(ss+k-1)/Z - truep(ss)*(ss-1)/Z - (1+np.log(Z)+np.log(1/k)+np.log(q))*pi_a;
    else:
        out = (-truep(0.25)*(0.25-1.1) + truep(ss+k)*(ss+k-1.1))/Z - np.log(Z)*a/Z*(ss+k-0.2)+a/Z*np.log(a/Z)*(0.2-ss)-(np.log(1/k)+np.log(q))*pi_a
    return out	


for tt in range(T):
    k = np.random.choice(range(K), 1, p = q)[0] #choose the arm  ;
    rr = np.random.uniform(0,1,1)
    xx[tt] = k/K + rr/K
    total[k] = total[k] + truep(xx[tt])/K
    nn[k] = nn[k] + 1
    sigma = explorefactor*np.sqrt(np.log(tt+1)/nn)
    q = (total/nn + sigma)/sum(total/nn + sigma)
    for k in range(K):
        ss = k/K;
        pi_a = func_pi_a(ss,1/K)
        KL[tt] = KL[tt] + func_KL(ss,1/K,q[k],pi_a)

        


import matplotlib
matplotlib.rcParams.update({'font.size': 20})

fig = plt.figure()
xxx = np.linspace(0,1,100)
yy = [truep(x) for x in xxx]
ax = fig.add_subplot(111)
Index = np.matrix.nonzero(q)
ax.plot(xxx,yy/Z,color='red')
plt.title("Daisee with K=%d" %K,fontsize=20)
for k in range(K):
    ss = k/K
    ax.plot((ss, ss+ M/K),(q[k]*K,q[k]*K),color='blue', linewidth=2.0)





    

    

