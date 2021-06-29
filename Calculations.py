'''
Author: Leo Goutte
Description: This is for continuum model's moire bands of TBG
'''

from numpy import *
import matplotlib.pyplot as plt
import numpy as np
# import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#############################################################################

# define parameters
d      = 2.46           #angstrom, whatever is ok.
hv     = 1.5*d*2970     #meV*angstrom, Fermi velocity for SLG
KDens  = 100            #density of k points, 100 is good.

# Imaginary unit
I      = complex(0, 1)

# callable function
def SpectrumDOSPlot(theta, tAA, tAB, valley, N, lim):
    """
    Creates fig object with spectrum and DOS of twisted bilayer graphene
    """
    # convert
    theta  *=np.pi/180.0 
    # lim = 3*tAA # y range of energies (3 * tAA is good)

    # global params
    MakeGlobalParameters(theta,valley)

    # layer tunneling matrices
    Tqb, Tqtr, Tqtl = MakeTunnelingMatrices(tAA,tAB)

    # make lattice
    Lattice(N)
    global siteN 
    siteN = (2*N+1)*(2*N+1)

    # diagonalize hamiltonian along path
    AllK, E = EnergiesOnPath(theta, valley, N)

    # trim energies
    Ks, Es = TrimEnergies(AllK,E,lim)

    # denisty of states
    eDOS, DOS = DensityOfStates(Es,lim)

    # get figure object
    fig = Figure(Ks,Es,DOS,eDOS,lim)

    return fig

# global params
def MakeGlobalParameters(theta,valley):

    global ei120
    global ei240

    ei120  = cos(2*pi/3) + valley*I*sin(2*pi/3)
    ei240  = cos(2*pi/3) - valley*I*sin(2*pi/3)

    global b1m
    global b2m
    global qb
    global K1
    global K2

    b1m    = 8*np.pi*sin(theta/2)/3/d*np.array([0.5, -np.sqrt(3)/2])
    b2m    = 8*np.pi*sin(theta/2)/3/d*np.array([0.5, np.sqrt(3)/2])
    qb     = 8*np.pi*sin(theta/2)/3/sqrt(3)/d*array([0, -1])
    K1     = 8*np.pi*sin(theta/2)/3/sqrt(3)/d*array([-sqrt(3)/2,-0.5])
    K2     = 8*np.pi*sin(theta/2)/3/sqrt(3)/d*array([-sqrt(3)/2,0.5])

# matrices in sublattice space
def MakeTunnelingMatrices(u1,u2):
    global Tqb
    global Tqtr
    global Tqtl

    Tqb    = np.array([[u1,u2], [u2,u1]], dtype=complex)
    Tqtr   = np.array([[u1*ei120, u2], [u2*ei240, u1*ei120]], dtype=complex)
    Tqtl   = np.array([[u1*ei240, u2], [u2*ei120, u1*ei240]], dtype=complex)

    return Tqb, Tqtr, Tqtl

#define Lattice
def Lattice(n):
    global L 
    global invL

    L = []
    invL = np.zeros((2*n+1, 2*n+1), int)
    
    count = 0
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])
            invL[i+n, j+n] = count
            count += 1
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])
            
    L = np.array(L)

# 2x2 dirac hamiltonian for given layer
def DiracHamiltonian(qx,qy,hv,valley):
    DH = np.zeros((2,2), dtype=complex)
    DH[0,1] = hv * (valley*qx - I*qy)
    DH[1,0] = hv * (valley*qx + I*qy)

    return DH

# workhorse
# energy for given momentum (heart of code)
def Energies(theta, kx, ky, valley, N):
    H = array(zeros((4*siteN, 4*siteN)), dtype=complex)
    for i in np.arange(siteN):
        #diagonal term
        ix = L[i, 0]
        iy = L[i, 1]
        ax = kx - valley*K1[0] + ix*b1m[0] + iy*b2m[0]
        ay = ky - valley*K1[1] + ix*b1m[1] + iy*b2m[1]

        qx = cos(theta/2) * ax + sin(theta/2) * ay
        qy =-sin(theta/2) * ax + cos(theta/2) * ay
         
        H[2*i:2*i+2, 2*i:2*i+2] = DiracHamiltonian(qx,qy,hv,valley)

        #off-diagonal term
        j = i + siteN

        # bottom hopping
        H[2*j:2*j+2, 2*i:2*i+2] = Tqb.T.conj()

        # top right hopping
        if (iy != valley*N): # check not on edge
            j = invL[ix+N, iy+valley*1+N] + siteN
            H[2*j:2*j+2, 2*i:2*i+2] = Tqtr.T.conj()

        # top left hopping
        if (ix != -valley*N): # check not on edge
            j = invL[ix-valley*1+N, iy+N] + siteN
            H[2*j:2*j+2, 2*i:2*i+2] = Tqtl.T.conj()
        

    for i in np.arange(siteN, 2*siteN):
        #diagonal term
        j = i - siteN
        ix = L[j, 0]
        iy = L[j, 1]
        ax = kx  - valley*K2[0] + ix*b1m[0] + iy*b2m[0] 
        ay = ky  - valley*K2[1] + ix*b1m[1] + iy*b2m[1]

        qx = cos(theta/2) * ax - sin(theta/2) * ay
        qy = sin(theta/2) * ax + cos(theta/2) * ay

        H[2*i:2*i+2, 2*i:2*i+2] = DiracHamiltonian(qx,qy,hv,valley)

        #off-diagonal term
        H[2*j:2*j+2, 2*i:2*i+2] = Tqb
        if (iy != -valley*N):
            j = invL[ix+N, iy-valley*1+N]
            H[2*j:2*j+2, 2*i:2*i+2] = Tqtr
        if (ix != valley*N):
            j = invL[ix+valley*1+N, iy+N]
            H[2*j:2*j+2, 2*i:2*i+2] = Tqtl     
    eigenvalue,featurevector=np.linalg.eigh(H)
    eig_vals_sorted = np.sort(eigenvalue)
    E=eig_vals_sorted
    return E

# diagonalize along path
def EnergiesOnPath(theta, valley, N):
    # ktheta
    kt = -qb[1]

    # make path
    global AtoB
    global BtoC
    global CtoD
    global DtoA

    AtoB = np.arange(-1/2, 1/2, 1/KDens)
    BtoC = np.arange(-1, 0, 1/KDens)
    CtoD = np.arange(0, sqrt(3), 1/KDens)
    DtoA = np.arange(0, 1, 1/KDens)
    AllK  = len(AtoB) + len(BtoC) + len(CtoD) + len(DtoA)

    # energies
    E  = np.zeros((AllK,4*siteN), float)

    for i in range(0, len(AtoB)):
        k = AtoB[i]
        E[i] = Energies(theta, sqrt(3)/2*kt, k*kt, valley, N)
    for i in range(len(AtoB), len(AtoB)+len(BtoC)):
        k = BtoC[i-len(AtoB)]
        E[i] = Energies(theta, 0, k*kt, valley, N)
    for i in range(len(AtoB)+len(BtoC), len(AtoB)+len(BtoC)+len(CtoD)):
        k = CtoD[i-len(AtoB)-len(BtoC)]
        E[i] = Energies(theta, 1.0/2*k*kt, -k*sqrt(3)/2*kt, valley, N)
    for i in range(len(AtoB)+len(BtoC)+len(CtoD), AllK):
        k = DtoA[i-len(AtoB)-len(BtoC)-len(CtoD)]
        E[i] = Energies(theta, -sqrt(3)/2*k*kt, -1/2*k*kt, valley, N)
        
    return AllK, E

# trim energies
def TrimEnergies(K, E, lim):
    """
    Trims energies and respective K points
    Assumes limit is (-lim,lim)
    Dimension of E is (K, # of bands)
    """
    # tune
    Bands = E.shape[1]
    lim += 50
    
    # array of Ks (each row is copies of a given K)
    Ks = np.stack([np.arange(K) for _ in range(Bands)]).T
    
    # test if each band is included within the range
    idx = np.zeros(Bands,dtype=bool)
    for i in range(Bands):
        idx[i] = np.logical_and(np.all(E[:,i]<lim),np.all(E[:,i]>-lim))

    # make new E and K arrays
    E = E[:,idx]
    Ks = Ks[:,idx]
    
    return Ks, E

# normalized lorentzian for DOS calculation
def lorentzian(x,x0,gam):
    return gam**2 / ( gam**2 + ( x - x0 )**2)

# DOS
def DensityOfStates(E, lim):
    """
    DOS calculation from delta definition
    """
    # make energy 1d list
    Es=E.reshape(-1)

    # energy precision
    tol=10e-2
    size_energies=10000+1
    energies=np.linspace(-lim,lim,num=size_energies)

    DOS=np.zeros(size_energies,dtype='float')
    for E0 in Es:
        delta=lorentzian(energies,E0,tol)
        DOS+=delta
        
    return energies, DOS

# figure
# def Figure(AllK,E,DOS,eDOS):
    # fig,ax=plt.subplots(nrows=1,ncols=2)
    # for j in range(0,4*siteN):
    #     ax[0].plot(np.arange(AllK), E[:,j], linestyle="-", linewidth=2)
    # ax[0].set_xlim(0, AllK)
    # ax[0].set_ylim(-300,300)
    # ax[0].set_xticks([0, len(AtoB), len(AtoB)+len(BtoC), len(AtoB)+len(BtoC)+len(CtoD), AllK])
    # ax[0].set_xticklabels(["K'", "K", '$\Gamma$', '$\Gamma$',"K'"], fontsize=20)
    # # plt.yticks(fontsize=13)
    # ax[0].set_ylabel('E(meV)', fontsize=20)

    # ax[1].plot(DOS,eDOS)#,basefmt=" ",markerfmt=" ")
    # ax[1].set_yticks([])
    # ax[1].set_xticks([])
    # ax[1].set_xlim(1,max(DOS)+10)
    # ax[1].set_xlabel("Arb. units")

    # # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.01)


    # fig = px.scatter(x=DOS, y=eDOS)

    # return fig

def Figure(Ks,Es,DOS,eDOS,lim):
    """
    Figure object for App.
    Ks,Es have dimensions (# of k-points, # of bands)
    """
    lim -= 35
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01)

    for j in range(Es.shape[1]):
        fig.add_trace(go.Scatter(x=Ks[:,j], y=Es[:,j],
                                 mode='lines',
                                 name='lines'),
                                 row=1, col=1)
    fig.add_trace(go.Scatter(x=DOS,y=eDOS,fill='tozerox',fillcolor='DarkSlateGrey',marker=dict(size=1,color='DarkSlateGrey')),row=1, col=2)


    fig.update_yaxes(title_text="Energies [meV]", range=[-lim, lim], row=1, col=1)
    fig.update_xaxes(title_text="Momentum [Kθ]", 
    tickmode="array", 
    tickvals=[0,KDens,2*KDens,(2+np.sqrt(3))*KDens,(3+np.sqrt(3))*KDens-1], 
    ticktext=["K'","K","Γ","Γ","K'"],
    row=1, col=1)

    fig.update_yaxes(range=[-lim, lim], row=1, col=2)
    fig.update_xaxes(title_text="LDOS [arb. units]", tickmode="array", tickvals=[], ticktext=[], row=1, col=2)


    fig.update_layout(height=600, 
    width=800, 
    title={
        'text':'Spectrum and local density of states (LDOS)',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)

    return fig


