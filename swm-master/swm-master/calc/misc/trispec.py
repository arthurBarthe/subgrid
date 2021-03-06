## COMPUTE TRISPEC
from __future__ import print_function
path = '/home/mkloewer/python/swm/'
import os; os.chdir(path) # change working directory
import numpy as np
from scipy import sparse
import time as tictoc
from netCDF4 import Dataset
import glob
import matplotlib.pyplot as plt

# OPTIONS
runfolder = [3]
print('Calculating 3D spectrogramms from run ' + str(runfolder))

##
def trispec(a,dt,dx,dy):
    """ Computes a wavenumber-frequency plot for 3D (t,x,y) data via radial (k = sqrt(kx**2 + ky**2)) integration. TODO: correct normalisation, so that the integral in normal space corresponds to the integral in Fourier space.
    """
    
    nt,ny,nx = np.shape(a)
    kx = (1/(dx))*np.hstack((np.arange(0,(nx+1)/2.),np.arange(-nx/2.+1,0)))/float(nx)
    ky = (1/(dy))*np.hstack((np.arange(0,(ny+1)/2.),np.arange(-ny/2.+1,0)))/float(ny)
    f = (1/(dt))*np.hstack((np.arange(0,(nt+1)/2.),np.arange(-nt/2.+1,0)))/float(nt)

    kxx,kyy = np.meshgrid(kx,ky)
    # radial distance from kx,ky = 0
    kk = np.sqrt(kxx**2 + kyy**2) 

    if nx >= ny: #kill negative wavenumbers
        k  = kx[:int(nx/2)+1]
    else:
        k  = ky[:int(ny/2)+1]

    f = f[:int(nt/2)+1] #kill negative frequencies
    dk = k[1] - k[0]

    # create radial coordinates, associated with k[i]
    # nearest point interpolation to get points within the -.5,.5 annulus
    rcoords = []
    for i in range(len(k)):
        rcoords.append(np.where((kk>(k[i]-.5*dk))*(kk<=(k[i]+.5*dk))))

    # 3D FFT
    p = np.fft.fftn(a)
    p = np.real(p * np.conj(p))

    # mulitply by dk to have the corresponding integral
    spec = np.zeros((len(k),len(f)))
    for i in range(len(k)):
        spec[i,:] = np.sum(p[:int(nt/2)+1,rcoords[i][0],rcoords[i][1]],axis=1)*dk

    return k,f,spec

## read data
for r,i in zip(runfolder,range(len(runfolder))):
    runpath = path+'data/run%04i' % r
    
    if i == 0:
        #u = np.load(runpath+'/u_sub.npy')
        #v = np.load(runpath+'/v_sub.npy')
        h = np.load(runpath+'/h_sub.npy')        
        time = np.load(runpath+'/t_sub.npy')
        print('run %i read.' % r)

    else:
        #u = np.concatenate((u,np.load(runpath+'/u_sub.npy')))
        #v = np.concatenate((v,np.load(runpath+'/v_sub.npy')))
        h = np.concatenate((h,np.load(runpath+'/h_sub.npy')))
        time = np.hstack((time,np.load(runpath+'/t_sub.npy')))
        print('run %i read.' % r)

t = time / 3600. / 24.  # in days
tlen = len(time)
dt = time[1] - time[0]

## read param
global param
param = np.load(runpath+'/param.npy').all()

##
k,f,spec3d = trispec(h,dt,param['dx'],param['dy'])


## rossby radius
Ro_min = param['c_phase']/f_T.max()
Ro_max = param['c_phase']/f_T.min()
# dispersion relation
Ro_disp_min = param['beta']*k/(k**2 + 1./Ro_min**2)
Ro_disp_max = param['beta']*k/(k**2 + 1./Ro_max**2)


## plotting 3D
sd = 3600.*24.

fig,ax = plt.subplots(1,1)
c1 = ax.contourf(k[1:]*1e3,f[1:]*sd,np.log10(spec3d[1:,1:].T),64,cmap='viridis')
ax.loglog()

plt.colorbar(c1,ax=ax)

ax.plot(Ro_min*np.ones(2)*1e3,[1./f[-1]/sd,1/f[1]/sd],'w')
ax.plot(Ro_max*np.ones(2)*1e3,[1./f[-1]/sd,1/f[1]/sd],'w')

ax.plot(1./k[-1]*np.ones(2)/1e3,[1./f_T.min()/3600./24.,1./f_T.min()/3600./24.],'k')
ax.plot(1./k[-1]*np.ones(2)/1e3,[1./f_T.max()/3600./24.,1./f_T.max()/3600./24.],'k')


ax.plot(1./k[1:]/1e3,1./Ro_disp_min[1:]/3600./24.,'k')
ax.plot(1./k[1:]/1e3,1./Ro_disp_max[1:]/3600./24.,'k')

ax.set_xlim(1./k[-1]/1e3,1./k[1]/1e3)
ax.set_ylim(1./f[-1],1./f[1])

ax.set_xlabel('wavelength [km]')
ax.set_ylabel('period [days]')
ax.set_title('3D power spectrum of $\eta$')
plt.tight_layout()
plt.show()