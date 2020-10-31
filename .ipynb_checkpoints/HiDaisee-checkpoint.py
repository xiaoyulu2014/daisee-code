import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.cm as cm
import time
import copy

a = 0.5
def truep(x):
    return  (x>=0.25)*np.exp(-10*(1-x)) + (x<0.25)*a;

Z = 0.1 - 0.1*np.exp(-7.5) + 0.25*a;

             

def func_pi_a(ss,ll):
    if ss+1/2**ll < 0.25:
        out = a/Z/2**ll
    elif ss>0.25:
        out = 0.1/Z*(truep(ss+1/2**ll)-truep(ss))
    else:
        out = a/Z*(0.25-ss) + 0.1/Z*(truep(ss+1/2**ll)-truep(0.25));
    return out

def func_KL(ss,ll,q,pi_a):
    if ss+1/2**ll < 0.25:
        out = (np.log(a/Z)-np.log(2**ll)-np.log(q))*pi_a
    elif ss>0.25:
        out = truep(ss+1/2**ll)*(ss+1/2**ll-1)/Z - truep(ss)*(ss-1)/Z - (1+np.log(Z)+np.log(2**ll)+np.log(q))*pi_a;
    else:
        out = (-truep(0.25)*(0.25-1.1) + truep(ss+1/2**ll)*(ss+1/2**ll-1.1))/Z - np.log(Z)*a/Z*(ss+1/2**ll-0.2)+a/Z*np.log(a/Z)*(0.2-ss)-(np.log(2**ll)+np.log(q))*pi_a
    return out	


explorefactor = 1e-2
threshold = 50
M = 1

T = 100000
L_max = 100000
total = np.zeros(L_max)        #total is an estimate of Z: sum(p/u)
total2 = np.zeros(L_max)        #total is an estimate of Z: sum(p/u)
nn = np.zeros(L_max)
Y_min = np.empty(L_max)
Y_min[:] = np.nan
Y_max = np.empty(L_max)
Y_max[:] = np.nan
KL = np.zeros(L_max)
dd = np.ones(L_max)  #indicator of whether a parent/child node, initially all are parent nodes
mm = np.zeros(L_max)
pi_a = np.zeros(L_max)

timer1 = np.zeros(T)
timer2 = np.zeros(T)
timer3 = np.zeros(L_max)
xx = np.zeros(T)
pp = np.zeros(T)
maxlevel = 1
RR = np.zeros(T)
q = np.zeros(L_max)
q[0] = 1;
children = [0]
num_partitions = np.zeros(T)


for tt in range(T):
    ii=0
    ll=0
    ss=0
    ii = int(ii)
    ii = np.random.choice(np.array(children), 1, p = q[children])[0] #choose the arm 
    old_ii = copy.copy(ii)
    ll = np.floor(np.log2(ii+1))
    if ii==0:
        ss = 0
    else:
        ss = (ii-2**ll+1)/2**ll
    rr = np.random.uniform(0,1,1)
    xx[tt] = ss + M*rr/(np.power(2,ll))
    pp[tt] = truep(xx[tt])  
    v = pp[tt]/(2**ll)
    Y_min[ii] = np.nanmin((Y_min[ii],v))
    Y_max[ii] = np.nanmax((Y_max[ii],v))
    total[ii] = total[ii] +  v    
    total2[ii] = total2[ii] + v**2
    nn[ii] = nn[ii] + 1   
    ii = int(2*ii + (rr > 0.5) + 1)
    ll= ll + 1
    v = pp[tt]/(2**ll)
    total[ii] = total[ii] +  v    
    total2[ii] = total2[ii] + v**2
    nn[ii] = nn[ii] + 1      
    ii = copy.copy(old_ii) 
    num_partitions[tt] = len(children)
    if (nn[ii] > threshold):
        Ymin = Y_min[ii]/(Y_min[ii]+Y_max[ii])
        Ymax = Y_max[ii]/(Y_min[ii]+Y_max[ii])
        if Ymin*np.log(2*Ymin) + Ymax*np.log(2*Ymax) > np.log(20/17):
            children.remove(ii)
            children.append(2*ii+1)
            children.append(2*ii+2)  
            q[children] = total[children]/nn[children] + explorefactor*np.sqrt(np.log(tt+1)/nn[children])
            q[children] = q[children]/np.sum(q[children])  

# +
fig = plt.figure()
xxx = np.linspace(0,1,100)
yy = [truep(x) for x in xxx]
ax = fig.add_subplot(111)
Index = np.matrix.nonzero(q)
ax.plot(xxx,yy/Z,color='red')

tmp = []
indices = []
for i in children:
    ll = np.floor(np.log2(i+1))
    tmp.append(q[i])
    ss = (i+1-2**ll)/2**ll*(M)
    ax.plot((ss, ss+ M/(2**ll)),(q[i]*2**ll,q[i]*2**ll),color='blue')
# -

matplotlib.rcParams.update({'font.size': 12})

plt.plot(num_partitions,linewidth=2.0)
plt.title("number of partitions vs iterations")
plt.xlabel("iteration")
plt.ylabel("number of partitions")



