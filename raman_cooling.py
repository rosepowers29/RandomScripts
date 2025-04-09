import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.linalg as LA

#GLOBAL VARS
gamma=1.
eta = 0.15
omega = 10
Omega_R = 0.1

#FUNCTION DEFINITIONS

def rho_prime(rho_1,rho_2,g,n,dn):
    #rho is a vector of motional states
    R_opt_0 = ((gamma*(Omega_R/2)**2)/((0.5*(gamma**2)*(2*n+1)**2+(omega*(dn))**2)))
    R_opt_min1 = ((gamma*(Omega_R/2)**2)/((0.5*(gamma**2)*(2*n+1)**2+(omega*(dn-1))**2)))
    R_opt_plus1 = ((gamma*(Omega_R/2)**2)/((0.5*(gamma**2)*(2*n+1)**2+(omega*(dn+1))**2)))

    rho_dot=0.
    if n<7 and g==2:
        if n<6:
            term1 = -gamma*((eta**2)*(1+(2*n+1))+R_opt_0+(eta**2)*n*R_opt_min1+(eta**2)*(n+1)*R_opt_plus1)*rho_2[n]
        else:
            term1 = -gamma*((eta**2)*(1+(2*n+1))+R_opt_0+(eta**2)*n*R_opt_min1)*rho_2[n]
        term2 = R_opt_0*rho_1[n]
        if n>0:
            term3 = R_opt_min1*(eta**2)*n*rho_1[n-1]
        else:
            term3 = 0
        if n < 6:
            term4 = R_opt_plus1*(eta**2)*(n+1)*rho_1[n+1]
        else:
            term4 = 0
        rho_dot = term1+term2+term3+term4
    elif n<7 and g==1:
        if n<6:
            term1 = -gamma*(R_opt_0+n*(eta**2)*R_opt_min1+(n+1)*(eta**2)*R_opt_plus1)*rho_1[n]
        else:
            term1 = -gamma*(R_opt_0+n*(eta**2)*R_opt_min1)*rho_1[n]
        term2 = (1+R_opt_0)*rho_2[n]
        if n>0:
            term3 = n*(eta**2)*(1+R_opt_min1)*rho_2[n-1]
        else:
            term3 = 0
        if n<6:
            term4 = (n+1)*(eta**2)*(1+R_opt_plus1)*rho_2[n+1]
        else:
            term4 = 0

        rho_dot = term1+term2+term3+term4

    return(rho_dot)

    '''
    if n<7:
        term1 = (-gamma*(n+1)-R_opt_0*((1-2*eta**2)**2))*rho_2[n]
        term2 = (gamma*n+R_opt_0*((1-2*eta**2)**2))*rho_1[n]
        if n<6:
            term3 = -R_opt_plus1*(eta**2)*(rho_2[n+1]-rho_1[n+1])
        else:
            term3=0
        if n>0:
            term4 = -R_opt_min1*(eta**2)*(rho_2[n-1]-rho_1[n-1])
        else:
            term4=0
        
        if g==1:
            rho_dot = term1+term2+term3+term4
        elif g==2:
            rho_dot = -(term1+term2+term3+term4)

    return(rho_dot)
    '''
 

    
#initialize population in a thermal state
# rho = e^(-H/kT)/Tr[e^(-H/kT)]
# T = 2*hbar*omega/k (set hbar = 1)
def initialize_pop():
    rho=[]
    rho_weighted=0
    for n in range(7):
        H = n*omega
        rho_n = np.exp(-H/(2*omega))
        rho.append(rho_n)
        rho_weighted+=rho_n*n
    rho = np.array(rho)
    trace = np.sum(rho)
    print((rho_weighted)/trace)
    rho = rho/trace
    
    return(rho, rho_weighted/trace)

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
dt = 1
ts = np.linspace(0,5000,5000)

dns = [-1.0, -0.5, 0.0]
colors = plt.cm.YlGnBu(np.linspace(0.3,0.8,3))
plt.gca().set_prop_cycle(cycler('color',colors))
for dn in dns:
    #reinitialize populations for every dn
    rho_g1, rho_weighted = initialize_pop()
    rho_g2 = np.array([0.00]*7)
    n_avg = [rho_weighted]
    for t in ts[:-1]:
        rho_g1, rho_g2 = update_pop(rho_g1, rho_g2, dt, dn)
        weighted_avg = 0
        for n in range(7):
            weighted_avg += n*rho_g1[n]
        weighted_avg = weighted_avg/np.sum(rho_g1)
        n_avg.append(weighted_avg)
    plt.plot(ts, n_avg, label = "$\\delta n=$"+str(dn))
plt.xlabel("time [arb units]")
plt.ylabel("$\\langle n \\rangle$")
plt.legend()
plt.savefig("n_avgs.pdf")
plt.close()