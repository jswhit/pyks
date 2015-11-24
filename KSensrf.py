"""Ensemble square-root filters for the 1-d Kuramoto-Sivashinsky eqn"""
import numpy as np
import sys
from KS import KS
from enkf import serial_ensrf, bulk_ensrf, etkf, letkf, etkf_modens,\
                 serial_ensrf_modens, bulk_enkf

if len(sys.argv) < 3:
    msg="""python KSensrf.py covinflate covlocal method

all variables are observed, assimilation interval is 0.4,
8 ensemble members, observation error standard deviation = 0.01,
observation operator is 5 pt boxcar running mean.
Model parameters: dt=dtassim=0.1,diffusion=0.05,exponent=16,N=128.

time mean error and spread stats printed to standard output.

covlocal:  localization distance (distance at which Gaspari-Cohn polynomial goes
to zero).

method:  =0 for serial Potter method
         =1 for bulk Potter method (all obs at once)
         =2 for ETKF (no localization applied)
         =3 for LETKF (using observation localization)
         =4 for serial Potter method with localization via modulation ensemble
         =5 for ETKF with modulation ensemble
         =6 for serial Potter method using sqrt of localized Pb ensemble
         =7 for bulk EnKF (all obs at once) with perturbed obs.

covinflate1,covinflate2:  (optional) inflation parameters corresponding
to a and b in Hodyss and Campbell.  If not specified, a=b=1."""
    raise SystemExit(msg)

corrl = float(sys.argv[1])
method = int(sys.argv[2])
covinflate1=1.; covinflate2=1.
if len(sys.argv) > 3:
    covinflate1 = float(sys.argv[3])
    covinflate2 = float(sys.argv[4])

ntstart = 1000 # time steps to spin up truth run
ntimes = 11000 # ob times
nens = 20 # ensemble members
oberrstdev = 0.01; oberrvar = oberrstdev**2 # ob error
verbose = False # print error stats every time if True
dtassim = 0.3  # assimilation interval
smooth_len = 10 # smoothing interval for H operator (0 or identity obs).
gaussian = False # Gaussian or running average smoothing in H.
thresh = 0.99 # threshold for modulated ensemble eigenvalue truncation.
# other model parameters...
dt = 0.1; diffusion = 0.05; exponent = 16; npts = 128

np.random.seed(42) # fix random seed for reproducibility

# model instance for truth (nature) run
model = KS(N=npts,dt=dt,exponent=exponent,diffusion=diffusion)
# mode instance for ensemble
ensemble = KS(N=npts,members=nens,dt=dt,exponent=exponent,diffusion=diffusion)
for nt in range(ntstart): # spinup truth run
    model.advance()

# sample obs from truth, compute climo stats for model.
xx = []; tt = []
for nt in range(ntimes):
    model.advance()
    xx.append(model.x[0]) # single member
    tt.append(float(nt)*model.dt)
xtruth = np.array(xx,np.float)
timetruth = np.array(tt,np.float)
xtruth_mean = xtruth.mean()
xprime = xtruth - xtruth_mean
xvar = np.sum(xprime**2,axis=0)/(ntimes-1)
xtruth_stdev = np.sqrt(xvar.mean())
if verbose:
    print 'climo for truth run:'
    print 'x mean =',xtruth_mean
    print 'x stdev =',xtruth_stdev
# forward operator.
# identity obs.
ndim = ensemble.n
h = np.eye(ndim)
# smoothing in forward operator
# gaussian or heaviside kernel.
if smooth_len > 0:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i-j)
            if i-j < -(ndim/2): rr = float(ndim-j+i)
            if i-j > (ndim/2): rr = float(i-ndim-j)
            r = np.fabs(rr)/smooth_len
            if gaussian:
                h[j,i] = np.exp(-r**2/0.15) # Gaussian
            else: # running average (heaviside kernel)
                if r <= 1:
                    h[j,i] = 1.
                else:
                    h[j,i] = 0.
        # normalize H so sum of weight is 1
        h[j,:] = h[j,:]/h[j,:].sum()
obs = np.empty(xtruth.shape, xtruth.dtype)
for nt in range(xtruth.shape[0]):
    obs[nt] = np.dot(h,xtruth[nt])
obs = obs + oberrstdev*np.random.standard_normal(size=obs.shape)

# spinup ensemble
ntot = xtruth.shape[0]
nspinup = ntstart
for n in range(ntstart):
    ensemble.advance()

nsteps = int(dtassim/model.dt) # time steps in assimilation interval
if verbose:
    print 'ntstart, nspinup, ntot, nsteps =',ntstart,nspinup,ntot,nsteps
if nsteps % 1  != 0:
    raise ValueError, 'assimilation interval must be an integer number of model time steps'
else:
    nsteps = int(nsteps)

def ensrf(ensemble,xmean,xprime,h,obs,oberrvar,covlocal,method=1,z=None):
    if method == 0: # ensrf with obs one at time
        return serial_ensrf(xmean,xprime,h,obs,oberrvar,covlocal,covlocal)
    elif method == 1: # ensrf with all obs at once
        return bulk_ensrf(xmean,xprime,h,obs,oberrvar,covlocal)
    elif method == 2: # etkf (no localization)
        return etkf(xmean,xprime,h,obs,oberrvar)
    elif method == 3: # letkf
        return letkf(xmean,xprime,h,obs,oberrvar,covlocal)
    elif method == 4: # serial ensrf using 'modulated' ensemble
        return serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z)
    elif method == 5: # etkf using 'modulated' ensemble
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z)
    elif method == 6: # serial ensrf using sqrt of localized Pb
        return serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,None)
    elif method == 7: # enkf with perturbed obs all at once
        return bulk_enkf(xmean,xprime,h,obs,oberrvar,covlocal)
    else:
        raise ValueError('illegal value for enkf method flag')

# define localization matrix.
covlocal = np.eye(ndim)
if corrl < 2*ndim:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i-j)
            if i-j < -(ndim/2): rr = float(ndim-j+i)
            if i-j > (ndim/2): rr = float(i-ndim-j)
            r = np.fabs(rr)/corrl
            #if r < 1.: # Bohman taper
            #    taper = (1.-r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi
            #taper = np.exp(-(r**2/0.15)) # Gaussian
            # Gaspari-Cohn polynomial.
            rr = 2.*r
            taper = 0.
            if r <= 0.5:
                taper = ((( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2+1.
            elif r > 0.5 and r < 1.:
                taper = (((( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                        + 4.0 - 2.0 / (3.0 * rr)
            covlocal[j,i]=taper

# compute square root of covlocal
if method in [4,5]:
    evals, eigs = np.linalg.eigh(covlocal)
    evals = np.where(evals > 1.e-10, evals, 1.e-10)
    evalsum = evals.sum(); neig = 0
    frac = 0.0
    while frac < thresh:
        frac = evals[ndim-neig-1:ndim].sum()/evalsum
        neig += 1
    #print 'neig = ',neig
    zz = (eigs*np.sqrt(evals/frac)).T
    z = zz[ndim-neig:ndim,:]
    #import matplotlib.pyplot as plt
    #print zz[-1].min(),zz[-1].max(),np.sqrt(ndim)*eigs[:,-1].min(),np.sqrt(ndim)*eigs[:,-1].max()
    #print np.sqrt(evals[-1]/(ndim*frac))
    #scalefact_eig1 = zz[-1].max()
    #plt.plot(zz[-1]) # 1st eigenvector.
    #plt.show()
else:
    neig = 0
    z = None

# run assimilation.
fcsterr = []
fcstsprd = []
analerr = []
analsprd = []
diverged = False
fsprdmean = np.zeros(ndim,np.float)
asprdmean = np.zeros(ndim,np.float)
ferrmean = np.zeros(ndim,np.float)
aerrmean = np.zeros(ndim,np.float)
for nassim in range(0,ntot,nsteps):
    # assimilate obs
    xmean = ensemble.x.mean(axis=0)
    xmean_b = xmean.copy()
    xprime = ensemble.x - xmean
    # calculate background error, sprd stats.
    ferr = (xmean - xtruth[nassim])**2
    if np.isnan(ferr.mean()):
        diverged = True
        break
    fsprd = (xprime**2).sum(axis=0)/(ensemble.members-1)
    if nassim >= nspinup:
        fsprdmean = fsprdmean + fsprd
        ferrmean = ferrmean + ferr
        fcsterr.append(ferr.mean()); fcstsprd.append(fsprd.mean())
    # update state estimate.
    # use 'bulk' ensrf for first half of spinup period.
    #if nassim < nspinup/2:
    #    xmean,xprime =\
    #    ensrf(ensemble,xmean,xprime,h,obs[nassim,:],oberrvar,covlocal,method=1,z=z)
    #else:
    if 1:
        xmean,xprime =\
        ensrf(ensemble,xmean,xprime,h,obs[nassim,:],oberrvar,covlocal,method=method,z=z)
    # calculate analysis error, sprd stats.
    aerr = (xmean - xtruth[nassim])**2
    asprd = (xprime**2).sum(axis=0)/(ensemble.members-1)
    if nassim >= nspinup:
        asprdmean = asprdmean + asprd
        aerrmean = aerrmean + aerr
        analerr.append(aerr.mean()); analsprd.append(asprd.mean())
    if verbose:
        print nassim,timetruth[nassim],np.sqrt(ferr.mean()),np.sqrt(fsprd.mean()),np.sqrt(aerr.mean()),np.sqrt(asprd.mean())
    # relaxation to prior variance inflation
    inc = xmean - xmean_b
    inf_fact = np.sqrt(covinflate1 + \
    (asprd/fsprd**2)*((fsprd/ensemble.members) + covinflate2*(2.*inc**2/(ensemble.members-1))))
    if method == 2: inf_fact=1.0
    xprime *= inf_fact
    # run forecast model.
    ensemble.x = xmean + xprime
    for n in range(nsteps):
        ensemble.advance() # perfect model

# print out time mean stats.
# error and spread are normalized by observation error.
if diverged:
    print method,len(fcsterr),corrl,covinflate1,covinflate2,np.nan,np.nan,neig
else:
    fcsterr = np.array(fcsterr)
    fcstsprd = np.array(fcstsprd)
    analerr = np.array(analerr)
    analsprd = np.array(analsprd)
    fstdev = np.sqrt(fcstsprd.mean())
    astdev = np.sqrt(analsprd.mean())
    asprdmean = asprdmean/len(fcstsprd)
    aerrmean = aerrmean/len(analerr)
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(ndim),asprdmean,color='b',label='error')
    #plt.plot(np.arange(ndim),aerrmean,color='r',label='spread')
    #plt.legend()
    #plt.show()
    print method,len(fcsterr),corrl,covinflate1,covinflate2,oberrstdev,np.sqrt(fcsterr.mean()),fstdev,\
          np.sqrt(analerr.mean()),astdev,neig
