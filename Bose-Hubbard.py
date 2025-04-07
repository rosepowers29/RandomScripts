import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.linalg as LA
from scipy.integrate import tplquad

#GLOBAL VARS
E_r = 1 # MeV
#m = 8.1e4 # MeV
#hbar = 6.58e-22 # MeV * s
#k = np.sqrt(2*m/hbar**2) # units of Hz; enter x in units of 1/c
#c = 3.0e23 # fm/s
k=1
c=1
hbar=1
m=1

#FUNCTION DEFINITIONS
#define the matrix elements
def Hll(l, lp, q, V_lat):
    if abs(l-lp)==0:
        temp = E_r*(2*l+q)**2
        return(temp)
    elif abs(l-lp)==1:
        temp = (-1/4)*V_lat
        return(temp)
    else:
        return 0


#fill the matrix
#restrict l to [-5,5]
def fill_Hll(q, V_lat):
    Hll_mat = []
    Hll_mat.clear()
    for l in range(-5, 6):
        l_row = []
        l_row.clear()
        for lp in range(-5, 6):
            l_row.append(Hll(l,lp,q,V_lat))
        l_row = np.array(l_row)
        Hll_mat.append(l_row)
    Hll_mat = np.array(Hll_mat)
    return(Hll_mat)

#find the eigenvalues
def find_eigvals(Hll_matrix):
    return(np.linalg.eigvals(Hll_matrix))

#find the eigenvectors
def find_eigs(Hll_matrix):
    evals, evecs = LA.eigh(Hll_matrix)
    #eigvals, eigvecs = np.linalg.eig(Hll_matrix)

    #minVal1 = 1000
    #min_index1 = -1
    #sort the eigenvalues to find n=1
    #for i in range(len(eigvals)-1):
    #    if eigvals[i] < minVal1:
    #        minVal1 = eigvals[i]
    #        min_index1 = i
    #get the corresponding eigenvector
    return(evals,evecs)

#Fourier sum the eigs, get the Bloch wfn
def Bloch_wfn(eigvecs,x,q):
    eigvecs = np.array(eigvecs)
    l=np.arange(-5,6)
    expfacs = np.exp(2j*(x)*l)*np.exp(1j*q*x) 
    Bloch = np.sum(expfacs*eigvecs[:,0])
    return(Bloch)

def Bloch_wfn3d(eigvecs, x, y, z, q):
    Blochx = Bloch_wfn(eigvecs, x, q)
    Blochy = Bloch_wfn(eigvecs, y, q)
    Blochz = Bloch_wfn(eigvecs, z, q)
    return(Blochx*Blochy*Blochz)

#get the tunneling
def J(V_lat):
    min_matrix = fill_Hll(0,V_lat)
    max_matrix = fill_Hll(1,V_lat)
    min_evals = find_eigvals(min_matrix)
    max_evals = find_eigvals(max_matrix)
    #get the 1st band value
    E_min = min(min_evals)
    E_max = min(max_evals)
    J = (E_max - E_min)/4
    return(J)

#define Wannier function from Bloch fns
def W(x, qs, V_lat):
    W=0
    for q in qs:
        evals, eigvecs = find_eigs(fill_Hll(q,V_lat))
        Bloch = Bloch_wfn(eigvecs,x,q)
        W+=Bloch
    W = W/len(qs) #normalize
    return(W)
#Wannier 3d
#def W3d_4(x,y,z,V_lat):
    #find the eigvecs, Blochs separately and use Trapz
    W=0
    for q in qs:
        evals, eigvecs = find_eigs(fill_Hll(q,V_lat))
        Blochx = Bloch_wfn(eigvecs,x,q)
        Blochy = Bloch_wfn(eigvecs,y,q)
        Blochz = Bloch_wfn(eigvecs,z,q)
        Bloch = Blochx*Blochy*Blochz
        W+=Bloch
    W = W/len(qs)
    W = np.abs(W)**4
    return(W)

#define the potential
#def U(qs, V_lat,xlim,ylim,zlim):
    integ,error = tplquad(W3d_4, -xlim, xlim, -ylim, ylim, -zlim, zlim, args=(qs,V_lat))
    U=4*np.pi*integ
    return(U)
    
def U(xs, ys, zs, Ws):
    zint = np.trapz(Ws, zs)
    yint = np.trapz(zint, ys)
    xint = np.trapz(yint, xs)
    U = 4*np.pi*xint
    return(U)

##-----------------------------------------------------------------------------------------------------

#All independent variables
qs = np.linspace(-1, 1, 25)
xs_ = np.linspace(-1,1,10)
ys_ = np.linspace(-1,1,10)
zs_ = np.linspace(-1,1,10)
V_lats = np.linspace(0,50,100)

##-----------------------------------------------------------------------------------------------------
#INTERACTION ENERGY

#generate the W3ds
#make the eigenvectors, Blochs

eig_i=[]
for V_lat in V_lats:
    V_eigs = []
    for q in qs:
        Hllmat = fill_Hll(q, V_lat)
        evals, evecs = find_eigs(Hllmat)
        V_eigs.append(evecs)
    eig_i.append(V_eigs)
eig_i=np.array(eig_i)
print("done finding the eigenvectors!")


V_Blochs=[]
V_Blochs.clear()
for Vi in range(100):
    x_sums = []
    for x in xs_:
        y_sums = []
        for y in ys_:
            z_sums = []
            for z in zs_:
                sum=0
                for qi in range(25):
                    evecs = eig_i[Vi,qi]
                    Bloch3d = Bloch_wfn3d(evecs, x, y, z, q)
                    sum += Bloch3d
                sum = sum/len(qs)
                sum = (np.abs(sum))**4
                sum = sum/10000
                z_sums.append(sum)
            y_sums.append(z_sums)
        x_sums.append(y_sums)
    V_Blochs.append(x_sums)
        
#for each V_lat value, V_Blochs is a nested array but we haved summed out q-dependence
Us = []
for iV in range(len(V_lats)):
    B_array = V_Blochs[iV]
    Uv = U(xs_, ys_, zs_, B_array)
    Us.append(Uv)
plt.plot(V_lats, Us)
plt.grid(True, 'both')
plt.xlabel("$V_{lat}~[E_r]$")
plt.ylabel("Onsite Interaction $U~[E_r]$")
plt.savefig("interaction_energy_test.pdf")
plt.close()

##------------------------------------------------------------------------------------------------------
#BLOCH WAVEFUNCTIONS AND DENSITY
'''
#in the thesis, they do q=0 and q=1 only for the wfns
evals, eigvecs1 = find_eigs(fill_Hll(0,8))
evals, eigvecs2 = find_eigs(fill_Hll(1,8))



Bloch0 = []
Bloch1 = []
BlochDens0 = []
BlochDens1 = []
for x in xs_:

    B_temp = Bloch_wfn(eigvecs1,x,0)
    Bloch0.append(np.real(B_temp))
    BlochDens0.append((np.abs(B_temp))**2)
    func=0

    B_temp = Bloch_wfn(eigvecs2,x,1)
    Bloch1.append(np.real(B_temp))
    BlochDens1.append((np.abs(B_temp))**2)




plt.plot(xs_, Bloch0, color="green", label="$q~[\\hbar k]=0$")
plt.plot(xs_, Bloch1, color="red", label="$q~[\\hbar k]=1$")
plt.xlabel("$x/c ~(c~[fm/s])$")
plt.ylabel("$Re~[\\Phi (x)]$")
plt.title("$V_{lat}=8E_R,~n=1$")
plt.legend()
plt.savefig("Bloch_wfn.pdf")
plt.close()

plt.plot(xs_, BlochDens0, color="green", label="$q~[\\hbar k]=0$")
plt.plot(xs_, BlochDens1, color="red", label="$q~[\\hbar k]=1$")
plt.xlabel("$x/c ~(c~[fm/s])$")
plt.ylabel("$|\\Phi (x)|^2$")
plt.title("$V_{lat}=8E_R,~n=1$")
plt.legend()
plt.savefig("Bloch_density.pdf")
plt.close()


##--------------------------------------------------------------------------------------------
#ENERGY BANDS

band1=[]
band2=[]
band3=[]
band4=[]

for q in qs:
    mat = fill_Hll(q, 4)
    evals = find_eigvals(mat)
    evals.sort()
    band1.append(evals[0])
    band2.append(evals[1])
    band3.append(evals[2])
    band4.append(evals[3])

plt.close()
colors = plt.cm.cool(np.linspace(0,1,4))
plt.gca().set_prop_cycle(cycler('color',colors))
plt.plot(qs,band1)
plt.plot(qs,band2)
plt.plot(qs,band3)
plt.plot(qs,band4)
plt.xlabel("$q~[\\hbar k]$")
plt.ylabel("$E_q~[E_R]$")
plt.title("$V_{lat}=4E_R$")
plt.savefig("vlat4E_R.pdf")
'''
##--------------------------------------------------------------------------------------------
#TUNNELING POTENTIAL AND POTENTIAL RATIOS
Js = []
for V_lat in V_lats:
    Js.append(J(V_lat))

plt.plot(V_lats, Js, color="blue")
plt.yscale('log')
plt.grid(True, 'both')
plt.xlabel("$V_{lat}~[E_r]$")
plt.ylabel("Tunneling $J~[E_r]$")
plt.savefig("tunneling_vs_Vlat.pdf")
plt.close()

Js=np.array(Js)
Us=np.array(Us)
plt.plot(V_lats, Us/Js, color="blue")
plt.yscale('log')
plt.grid(True, 'both')
plt.xlabel("$V_{lat}~[E_r]$")
plt.ylabel("Ratio $U/J$")
plt.savefig("potential_ratios.pdf")
plt.close()


##-------------------------------------------------------------------------------------------
#WANNIER FUNCTIONS
'''
Ws=[]
W_dens=[]
for x in xs_:
    Wan = W(x, qs, 3)
    Ws.append(np.real(Wan))
    W_dens.append((np.abs(Wan))**2)

Vls=[]
for x in xs_:
    Vl = (np.sin((x))/2)**2 #dividing to match the arbitrary one in the thesis
    Vls.append(Vl)
#plt.plot(xs_,Ws,color="red",label="$w_0(x)$")
plt.plot(xs_, W_dens, color="red", label="density")
plt.plot(xs_, Vls, color="blue", label="Arbitrary $V_{lat}(x)$")
plt.axhline(y=0,color="grey")
plt.xlabel("$x/c ~(c~[fm/s])$")
#plt.ylabel("$Re~[w_0 (x)]$")
plt.ylabel("$|w_0 (x)|^2$")
plt.title("$E_R = 3$")
plt.legend()
plt.savefig("Wannier_dens.pdf")
plt.close()

'''


