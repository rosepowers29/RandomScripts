import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.linalg as LA

#GLOBAL VARS
gamma=1.
gamm_nn=1
eta = 0.15
omega = 10
Omega_R = 0.1

#FUNCTION DEFINITIONS
def rho_prime(rho_1,rho_2,g,n,dn):
    #rho is a vector of states
    R_opt = ((gamma*Omega_R**2)/(2*(gamma**2+(omega**2)*(dn)**2)))
    rho_dot=0.
    if n < 7:
        term1 = rho_2[n]*(-gamma-R_opt*(1-2*eta**2)**2)
        term2 = rho_1[n]*R_opt*(1-2*eta**2)**2
        if n > 0:
            term3 = (R_opt*eta**2)*(rho_1[n-1]-rho_2[n-1])
        else:
            term3 = 0.0
        if n < 6:
            term4 = (R_opt*eta**2)*(rho_1[n+1]-rho_2[n+1])
        else:
            term4=0.0
        if g == 2:
            rho_dot = term1+term2+term3+term4
        elif g == 1:
            rho_dot = -(term1+term2+term3+term4)
    return(rho_dot)
    
    '''
    if g == 1 and n <7:
        term1=gamma*rho_2[n]
        if n>0:
            term3=(eta_op**2)*n*rho_2[n-1]
            term5=-0.5*R_opt*(eta_R**2)*n*(rho_1[n]-rho_2[n-1])
        else:
            term3=0
            term5=0
        term4=-0.5*R_opt*(rho_1[n]-rho_2[n])
        if n<6:
            term2=gamma*(eta_op**2)*(n+1)*rho_2[n+1]
            term6=-0.5*R_opt*(eta_R**2)*(n+1)*(rho_1[n]-rho_2[n+1])
        else:
            term2=0
            term6=0
        rho_dot = term1+term2+term3+term4+term5+term6
        #print(rho_dot)
        return(rho_dot)
    
    elif g == 2 and n<7:
        term2=0.5*R_opt*(rho_1[n]-rho_2[n])
        if n>0:
            term3=0.5*R_opt*n*(eta_R**2)*(rho_1[n-1]-rho_2[n])
        else:
            term3=0
        if n<6:
            term4=0.5*(n+1)*(eta_R**2)*R_opt*(rho_1[n+1]-rho_2[n])
            term1=-gamma*(1+(2*n+1)*(eta_op**2))*rho_2[n]
        else:
            term4=0
            term1=0
        rho_dot = term1+term2+term3+term4
        #print(rho_dot)
        return(rho_dot)
    else:
        return(0)
'''
    
#initialize population in a thermal state
# rho = e^(-H/kT)/Tr[e^(-H/kT)]
# T = 2*hbar*omega/k (set hbar = 1)
def initialize_pop():
    rho=[]
    for n in range(7):
        H = n*omega
        rho_n = np.exp(-H/(2*omega))
        rho.append(rho_n)
    rho = np.array(rho)
    trace = np.sum(rho)
    rho = rho/trace
    return(rho)

# let the state evolve
def update_pop(rho_1, rho_2, dt, dn):
    for n in range(7):
        d_rho1 = rho_prime(rho_1, rho_2, 1, n, dn)
        d_rho2 = rho_prime(rho_1, rho_2, 2, n, dn)
        rho_1[n] = rho_1[n]+d_rho1*dt
        rho_2[n] = rho_2[n]+d_rho2*dt
    #rho_1_norm = rho_1/np.sum(rho_1)
    #rho_2_norm = rho_2/np.sum(rho_2)
    return(rho_1, rho_2)

##-----------------------------------------------------------------------------------------------------------
dt = .1
ts = np.linspace(0,20,200)

#initialize our populations!
rho_g1 = initialize_pop()
rho_g2 = np.array([0.00]*7)


#evolve! start with dn = -1
rhos = [(rho_g1, rho_g2)]
rho1s=[rho_g1]
n_avg=[]
for t in ts:
    rho_g1, rho_g2 = update_pop(rho_g1, rho_g2, dt, -.5)
    rhos.append((rho_g1, rho_g2))
    rho1s.append(rho_g1)
    weighted_avg = 0
    for n in range(7):
        weighted_avg += rho_g1[n]*n
    #weighted_avg = weighted_avg
    n_avg.append(weighted_avg)

rho_arrays=[]
for n in range(7):
    n_array=[]
    for t in range(200):
        rho_t = rho1s[t]
        rho_t_n = rho_t[n]
        n_array.append(rho_t_n)
    rho_arrays.append(n_array)


plt.plot(ts, n_avg, color='blue')
#colors = plt.cm.spring(np.linspace(0,1,7))
#plt.gca().set_prop_cycle(cycler('color',colors))
#for i in range(7):
#    plt.plot(ts, rho_arrays[i],label="Fraction of pop. in state "+str(i))
plt.xlabel("time [s]")
plt.ylabel("$\\langle n \\rangle$")
plt.title("$\\delta n=-.5$")
#plt.legend()
plt.savefig("n_avg_carrier.pdf")
plt.close()