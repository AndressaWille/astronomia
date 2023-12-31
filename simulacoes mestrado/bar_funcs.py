## funções para análise de galáxias barradas

from numpy import *
import numpy as np
import h5py
import unsiotools.unsiotools.simulations.cfalcon as falcon


def A1_warp(m, x, y, z, Rmax, Nbins, n_snapshots):
    
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins

    r = np.empty(Nbins)
    a1 = np.empty(Nbins)
    b1 = np.empty(Nbins)
    A1 = np.empty(Nbins)

    for i in range(0, Nbins):

        R1 = i * dR
        R2 = R1 + dR
        r[i] = (0.5 * (R1+R2))

        over = 1.0 * dR

        cond = np.argwhere((R > R1-over) & (R < R2+over)).flatten()

        N = len(z[cond])
        
        a1[i] = np.sum(z[cond] * np.cos(1*theta[cond]))/N
        
        b1[i] = np.sum(z[cond] * np.sin(1*theta[cond]))/N

        A1[i] = np.sqrt(a1[i]**2+b1[i]**2)

        if math.isnan(A1[i]) == True:
            A1[i] = 0

    return A1, r



def density(m, x, y, z, Rmax, Nbins, n_snapshots):
    R = np.sqrt(x**2 + y**2 + z**2)
    
    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins

    r = np.empty(Nbins)
    M = np.empty(Nbins)
    V = np.empty(Nbins)
    densities = np.empty(Nbins)

    for i in range(0, Nbins):
        R1 = i * dR
        R2 = R1 + dR
        r[i] = (0.5 * (R1+R2))

        over = 1.0 * dR

        cond = np.argwhere((R > R1-over) & (R < R2+over)).flatten()
        M[i] = sum(m[cond] * 1e10)
        V2 = ((4.0/3.0) * np.pi * (R2**3))
        V1 = ((4.0/3.0) * np.pi * (R1**3))
        V[i] = V2 - V1
        densities[i] = M[i]/V[i]

    return densities, r
    

def density_sup(m, x, y, Rmax, Nbins, n_snapshots):
    R = np.sqrt(x**2 + y**2)
    
    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins

    r = np.empty(Nbins)
    M = np.empty(Nbins)
    A = np.empty(Nbins)
    densities = np.empty(Nbins)

    for i in range(0, Nbins):
        R1 = i * dR
        R2 = R1 + dR
        r[i] = (0.5 * (R1+R2))

        over = 1.0 * dR

        cond = np.argwhere((R > R1-over) & (R < R2+over)).flatten()
        M[i] = sum(m[cond] * 1e10)
        A[i] = np.pi * ((R2)**2 - (R1)**2)
        densities[i] = M[i]/A[i]

    return densities, r

    
def z_theta(x, y, z):
    theta_ = []
    z_ = []
    
    r = sqrt(x**2 + y**2)
    rmin = 20.0
    rmax = 30.0
    theta_disk = np.arctan2(y, x)
                
    theta_min = (-1) * np.pi
    theta_max = np.pi
    Nbins = 30
    dtheta = (theta_max - theta_min)/Nbins
                
    for i in range(-Nbins, Nbins):
        theta1 = i * dtheta
        theta2 = theta1 + dtheta
        theta_.append((theta1 + theta2) * 0.5)
        cond = argwhere((theta_disk>theta1) & (theta_disk<theta2) & (r>rmin) & (r<rmax)).flatten()
        zmean = mean(z[cond])
        z_.append(zmean)

    return theta_, z_

def find_peak(x, y, z, m):
    #build pos array
    pos = np.array( [x, y, z] )
    #format required by UNS
    pos = pos.T.astype('float32').flatten()
    #use falcon to get densities
    cf  = falcon.CFalcon()
    _, dens, _ = cf.getDensity(pos, m)
    
    # call function to compute densities
    dens = dens
    #sort particles by decreasing density
    sort = np.argsort(dens)[::-1]
    x = x[sort]
    y = y[sort]
    z = z[sort]
    dens = dens[sort]
    #select a few of the densest particles
    Ndensest = 64
    #average positions of the densest particles
    xpeak = np.average(x[0:Ndensest])
    ypeak = np.average(y[0:Ndensest])
    zpeak = np.average(z[0:Ndensest])
    
    return xpeak, ypeak, zpeak


def theta_bar(m, x, y, Rmax):
    
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    
    Rmax = Rmax

    cond = np.argwhere(R<Rmax).flatten()
    a2 = sum(m[cond] * np.cos(2*theta[cond]))
    b2 = sum(m[cond] * np.sin(2*theta[cond]))

    theta_b = (1/2)*np.arctan2(b2,a2)
    #theta_b_graus = theta_b *180/np.pi #graus

    return theta_b
    

def corr_theta_b(N, x, y, theta):
    
    for k in range (0, N):
        x_2 = np.cos(-theta)*x[k] - np.sin(-theta)*y[k]
        y_2 = np.sin(-theta)*x[k] + np.cos(-theta)*y[k]
        x[k] = x_2 
        y[k] = y_2 
        
    return x, y     

def theta_phi(N, x, y, z, vx, vy, vz, m):
    Ltot = [0, 0, 0]
    
    for k in range (0, N):
        R = [(x[k]), (y[k]), (z[k])]
        px = (m[k] * vx[k])
        py = (m[k] * vy[k])
        pz = (m[k] * vz[k])
        p = [px, py, pz]
        L = np.cross(R, p)
        Ltot = Ltot + L
        
    Lx = Ltot[0]
    Ly = Ltot[1]
    Lz = Ltot[2]
    
    theta = np.arctan2(Ly, Lx)
    
    Lxy = np.sqrt(Lx**2 + Ly**2)
    
    phi = np.arctan2(Lxy, Lz)
    
    return theta, phi
   
    
def angular_momentum_1(N, x, y, vx, vy, theta): 
        
    for k in range (0, N):
        x_2 = np.cos(-theta)*x[k] - np.sin(-theta)*y[k]
        y_2 = np.sin(-theta)*x[k] + np.cos(-theta)*y[k]
        x[k] = x_2 
        y[k] = y_2 
        
        vx_2 = np.cos(-theta)*vx[k] - np.sin(-theta)*vy[k]
        vy_2 = np.sin(-theta)*vx[k] + np.cos(-theta)*vy[k]
        vx[k] = vx_2 
        vy[k] = vy_2
        
    return x, y, vx, vy


def angular_momentum_2(N, x, z, vx, vz, phi):
    
    for k in range (0, N):
        z_2 = np.cos(-phi)*z[k] - np.sin(-phi)*x[k]
        x_3 = np.sin(-phi)*z[k] + np.cos(-phi)*x[k]
        z[k] = z_2
        x[k] = x_3 
         
        vz_2 = np.cos(-phi)*vz[k] - np.sin(-phi)*vx[k]
        vx_3 = np.sin(-phi)*vz[k] + np.cos(-phi)*vx[k]
        vz[k] = vz_2
        vx[k] = vx_3
        
    return x, z, vx, vz


def com(m, x, y, z):
    
    cm_x = sum(m*x)/sum(m)
    cm_y = sum(m*y)/sum(m)
    cm_z = sum(m*z)/sum(m)
    
    return cm_x, cm_y, cm_z

def shift_com(m, x, y, z):
    
    cm_x = sum(m*x)/sum(m)
    cm_y = sum(m*y)/sum(m)
    cm_z = sum(m*z)/sum(m)

    x_new = x - cm_x
    y_new = y - cm_y
    z_new = z - cm_z
    
    return x_new, y_new, z_new


def shift_com_2(m_disk, x_disk, y_disk, z_disk, m_halo, x_halo, y_halo, z_halo):

    cm_x_disk = sum(m_disk*x_disk)/sum(m_disk)
    cm_y_disk = sum(m_disk*y_disk)/sum(m_disk)
    cm_z_disk = sum(m_disk*z_disk)/sum(m_disk)

    x_new_disk = x_disk - cm_x_disk
    y_new_disk = y_disk - cm_y_disk
    z_new_disk = z_disk - cm_z_disk

    cm_x_halo = sum(m_halo*x_halo)/sum(m_halo)
    cm_y_halo = sum(m_halo*y_halo)/sum(m_halo)
    cm_z_halo = sum(m_halo*z_halo)/sum(m_halo)

    x_new_halo = x_halo - cm_x_halo
    y_new_halo = y_halo - cm_y_halo
    z_new_halo = z_halo - cm_z_halo
    
    return x_new_disk, y_new_disk, z_new_disk, x_new_halo, y_new_halo, z_new_halo


def bar_strength(m, x, y, Rmax, Nbins, n_snapshots):
    
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins

    r = np.empty(Nbins)
    ab = np.empty(Nbins)
    a0 = np.empty(Nbins)
    i2 = np.empty(Nbins)

    for i in range(0, Nbins):

        R1 = i * dR
        R2 = R1 + dR
        r[i] = (0.5 * (R1+R2))

        over = 1.0 * dR

        cond = np.argwhere((R > R1-over) & (R < R2+over)).flatten()

        a = m[cond] * np.cos(2*theta[cond])
        a_quad = sum(a)

        a_ = m[cond] * np.cos(0*theta[cond])
        a0[i] = sum(a_)

        b = m[cond] * np.sin(2*theta[cond])
        b_quad = sum(b)

        ab[i] = a_quad**2+b_quad**2

        i2[i] = np.sqrt(ab[i])/a0[i]
            

    a2 = max(i2)

    return a2


def v_circ(m_disk, m_halo, x_new_disk, y_new_disk, z_new_disk, x_new_halo, y_new_halo, z_new_halo, Rmax, Nbins):
    
    G = 43007.1

    R_disk = np.sqrt(x_new_disk**2 + y_new_disk**2 + z_new_disk**2)
    R_halo = np.sqrt(x_new_halo**2 + y_new_halo**2 + z_new_halo**2)

    M_r_disk = np.empty(Nbins)
    M_r_halo = np.empty(Nbins)
    M_r = np.empty(Nbins)
    r = np.empty(Nbins)
    v_c = np.empty(Nbins)
    v_c_disk = np.empty(Nbins)
    v_c_halo = np.empty(Nbins)

    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins
    
    for i in range(0, Nbins):
        R1 = i * dR
        R2 = R1 + dR
        r[i] = R2

        cond1 = np.argwhere(R_disk<=R2).flatten()
        M_r_disk[i] = sum(m_disk[cond1])

        cond2 = np.argwhere(R_halo<=R2).flatten()
        M_r_halo[i] = sum(m_halo[cond2])

        v_c_disk[i] = (np.sqrt(G*M_r_disk[i]/r[i]))
        v_c_halo[i] = (np.sqrt(G*M_r_halo[i]/r[i]))

        M_r[i] = (M_r_disk[i] + M_r_halo[i])

        v_c[i] = (np.sqrt(G*M_r[i]/r[i]))
    
    return v_c_disk, v_c_halo, v_c, r


def v_circ_comp(m, x, y, z, Rmax, Nbins):
    
    G = 43007.1
    R = np.sqrt(x**2 + y**2 + z**2)

    M_r = np.empty(Nbins)
    r = np.empty(Nbins)
    v_c = np.empty(Nbins)

    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins
    
    for i in range(0, Nbins):
        R1 = i * dR
        R2 = R1 + dR
        r[i] = R2

        cond = np.argwhere(R<=R2).flatten()
        M_r[i] = sum(m[cond])

        v_c[i] = (np.sqrt(G*M_r[i]/r[i]))
    
    return v_c, r, M_r


def S(m_up, x_up, y_up, m_down, x_down, y_down, Rmax, Nbins, n_snapshots):
    #S = |A2(z>0) - A2(z<0)|
    
    a2_up = bar_strength(m_up, x_up, y_up, Rmax, Nbins, n_snapshots)
    a2_down = bar_strength(m_down, x_down, y_down, Rmax, Nbins, n_snapshots)
    
    S = abs(a2_up - a2_down)
    
    return S


def time_buckling(S1, time1):
    maxS = max(S1) 
    pos_max = np.where(S1 == maxS)
    time_max = float(time1[pos_max])
    
    return time_max