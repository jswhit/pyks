import numpy as np

def symsqrtm(a):
    """symmetric square-root of a symmetric positive definite matrix"""
    evals, eigs = np.linalg.eigh(a)
    sqrtevals = np.sqrt(evals)
    symsqrt = np.dot(np.dot(eigs,np.diag(sqrtevals)),eigs.T)
    symsqrtinv = np.dot(np.dot(eigs,np.diag(1./sqrtevals)),eigs.T)
    return symsqrt, symsqrtinv

def serial_ensrf(xmean,xprime,h,obs,oberrvar,covlocal,obcovlocal):
    """serial potter method"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    #hxmean = np.dot(h,xmean)
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(ndim),xmean)
    #plt.plot(np.arange(ndim),hxmean)
    #plt.show()
    #raise SystemExit
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        hxprime = np.dot(xprime,h[nob])
        hxmean = np.dot(h[nob],xmean)
        # state space update
        hxens = hxprime.reshape((nanals, 1))
        hpbht = (hxens**2).sum()/(nanals-1)
        gainfact = ((hpbht+oberrvar)/hpbht*\
                   (1.-np.sqrt(oberrvar/(hpbht+oberrvar))))
        pbht = (xprime.T*hxens[:,0]).sum(axis=1)/float(nanals-1)
        kfgain = covlocal[nob,:]*pbht/(hpbht+oberrvar)
        xmean = xmean + kfgain*(ob-hxmean)
        xprime = xprime - gainfact*kfgain*hxens
    return xmean, xprime

def serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z):
    """serial potter method"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]

    # if True, use gain from modulated ensemble to
    # update perts.  if False, use gain from original ensemble.
    update_xprime = True
    if z is None:
        # set ensemble to square root of localized Pb
        Pb = covlocal*np.dot(xprime.T,xprime)/(nanals-1)
        evals, eigs = np.linalg.eigh(Pb)
        evals = np.where(evals > 1.e-10, evals, 1.e-10)
        nanals2 = eigs.shape[0]
        xprime2 = np.sqrt(nanals2-1)*(eigs*np.sqrt(evals)).T
    else:
        # modulation ensemble
        neig = z.shape[0]; nanals2 = neig*nanals; nanal2 = 0
        xprime2 = np.zeros((nanals2,ndim),xprime.dtype)
        for j in range(neig):
            for nanal in range(nanals):
                #print j,nanal,z[neig-j-1,:].min(), z[neig-j-1,:].max()
                xprime2[nanal2,:] = xprime[nanal,:]*z[neig-j-1,:]
                # unmodulated member is j=1, scaled by z[-1] (a constant)
                nanal2 += 1
        xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2
    #print xprime2.shape
    #print ((xprime**2).sum(axis=0)/(nanals-1)).sum()
    #print ((xprime2**2).sum(axis=0)/(nanals2-1)).sum()
    #raise SystemExit

    # update xmean using full xprime2
    # update original xprime using gain from full xprime2
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        hxprime = np.dot(xprime2,h[nob])
        hxprime_orig = np.dot(xprime,h[nob])
        hxmean = np.dot(h[nob],xmean)
        # state space update
        hxens = hxprime.reshape((nanals2, 1))
        hxens_orig = hxprime_orig.reshape((nanals, 1))
        hpbht = (hxens**2).sum()/(nanals2-1)
        gainfact = ((hpbht+oberrvar)/hpbht*\
                   (1.-np.sqrt(oberrvar/(hpbht+oberrvar))))
        pbht = (xprime2.T*hxens[:,0]).sum(axis=1)/float(nanals2-1)
        kfgain = pbht/(hpbht+oberrvar)
        xmean = xmean + kfgain*(ob-hxmean)
        xprime2 = xprime2 - gainfact*kfgain*hxens
        if not update_xprime:
            hpbht = (hxens_orig**2).sum()/(nanals-1)
            gainfact = ((hpbht+oberrvar)/hpbht*\
                       (1.-np.sqrt(oberrvar/(hpbht+oberrvar))))
            pbht = (xprime.T*hxens_orig[:,0]).sum(axis=1)/float(nanals-1)
            kfgain = covlocal[nob,:]*pbht/(hpbht+oberrvar)
        xprime  = xprime  - gainfact*kfgain*hxens_orig
    #print ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    #print ((xprime**2).sum(axis=0)/(nanals-1)).mean()
    #raise SystemExit
    return xmean, xprime

def bulk_ensrf(xmean,xprime,h,obs,oberrvar,covlocal):
    """bulk potter method"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    R = oberrvar*np.eye(nobs)
    Rsqrt = np.sqrt(oberrvar)*np.eye(nobs)
    Pb = np.dot(np.transpose(xprime),xprime)/(nanals-1)
    Pb = covlocal*Pb
    C = np.dot(np.dot(h,Pb),h.T)+R
    Cinv = np.linalg.inv(C)
    Csqrt, Csqrtinv =  symsqrtm(C)
    kfgain = np.dot(np.dot(Pb,h.T),Cinv)
    reducedgain = np.dot(np.dot(np.dot(Pb,h.T),Csqrtinv.T),np.linalg.inv(Csqrt + Rsqrt))
    xmean = xmean + np.dot(kfgain, obs-np.dot(h,xmean))
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal])
    hxprime_tmp = hxprime.reshape((nanals, ndim, 1))
    xprime = xprime - np.dot(reducedgain, hxprime_tmp).T.squeeze()
    return xmean, xprime

def bulk_enkf(xmean,xprime,h,obs,oberrvar,covlocal,denkf=False):
    """bulk enkf method with perturbed obs, or DEnKF"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    R = oberrvar*np.eye(nobs)
    Rsqrt = np.sqrt(oberrvar)*np.eye(nobs)
    Pb = np.dot(np.transpose(xprime),xprime)/(nanals-1)
    Pb = covlocal*Pb
    C = np.dot(np.dot(h,Pb),h.T)+R
    Cinv = np.linalg.inv(C)
    kfgain = np.dot(np.dot(Pb,h.T),Cinv)
    xmean = xmean + np.dot(kfgain, obs-np.dot(h,xmean))
    if not denkf:
        obnoise = np.sqrt(oberrvar)*np.random.standard_normal(size=(nanals,nobs))
        obnoise = obnoise - obnoise.mean(axis=0)
    else:
        obnoise = np.zeros((nanals,nobs))
        kfgain = 0.5*kfgain
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal]) + obnoise[nanal]
    hxprime_tmp = hxprime.reshape((nanals, ndim, 1))
    xprime = xprime - np.dot(kfgain, hxprime_tmp).T.squeeze()
    return xmean, xprime

def etkf(xmean,xprime,h,obs,oberrvar):
    """ETKF (use only with full rank ensemble, no localization)"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    # forward operator.
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal])
    hxmean = np.dot(h,xmean)
    Rinv = (1./oberrvar)*np.eye(nobs)
    YbRinv = np.dot(hxprime,Rinv)
    pa = (nanals-1)*np.eye(nanals)+np.dot(YbRinv,hxprime.T)
    evals, eigs = np.linalg.eigh(pa)
    # make square root symmetric.
    painv = np.dot(np.dot(eigs,np.diag(np.sqrt(1./evals))),eigs.T)
    kfgain = np.dot(xprime.T,np.dot(np.dot(painv,painv.T),YbRinv))
    enswts = np.sqrt(nanals-1)*painv
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    xprime = np.dot(enswts.T,xprime)
    return xmean, xprime

def etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z,denkf=False):
    """ETKF with modulated ensemble. Perturbed of or DEnKF for ens perts."""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    if z is None:
        raise ValueError('z not specified')
    # modulation ensemble
    neig = z.shape[0]; nanals2 = neig*nanals; nanal2 = 0
    xprime2 = np.zeros((nanals2,ndim),xprime.dtype)
    for j in range(neig):
        for nanal in range(nanals):
            xprime2[nanal2,:] = xprime[nanal,:]*z[neig-j-1,:]
            nanal2 += 1
    xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2
    # forward operator.
    hxprime = np.empty((nanals2, nobs), xprime2.dtype)
    for nanal in range(nanals2):
        hxprime[nanal] = np.dot(h,xprime2[nanal])
    hxmean = np.dot(h,xmean)
    Rinv = (1./oberrvar)*np.eye(nobs)
    YbRinv = np.dot(hxprime,Rinv)
    pa = (nanals2-1)*np.eye(nanals2)+np.dot(YbRinv,hxprime.T)
    evals, eigs = np.linalg.eigh(pa)
    painv = np.dot(np.dot(eigs,np.diag(1./evals)),eigs.T)
    kfgain = np.dot(xprime2.T,np.dot(painv,YbRinv))
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    # perturbed obs update of original ensemble
    if not denkf:
        obnoise = np.sqrt(oberrvar)*np.random.standard_normal(size=(nanals,nobs))
        obnoise = obnoise - obnoise.mean(axis=0)
    else:
        obnoise = np.zeros((nanals,nobs))
        kfgain = 0.5*kfgain
        #R = oberrvar*np.eye(nobs)
        #Rsqrt = np.sqrt(oberrvar)*np.eye(nobs)
        #hpbht = np.dot(hxprime.T, hxprime)
        #C = hpbht+R; Cinv = np.linalg.inv(C)
        ##pbht = np.dot(xprime2.T, hxprime)
        #pbht = np.dot(kfgain, C)
        #Csqrt, Csqrtinv =  symsqrtm(C)
        #kfgain = np.dot(np.dot(pbht,Csqrtinv.T),np.linalg.inv(Csqrt + Rsqrt))
    #oberr = np.sqrt((obnoise**2).sum(axis=0)/(nanals-1))
    #print oberr.mean(), np.sqrt(oberrvar)
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal]) + obnoise[nanal]
    hxprime_tmp = hxprime.reshape((nanals, ndim, 1))
    xprime = xprime - np.dot(kfgain, hxprime_tmp).T.squeeze()
    # random sample of modulated analysis ensemble.
    #enswts = np.sqrt(nanals2-1)*np.dot(np.dot(eigs,np.diag(np.sqrt(1./evals))),eigs.T)
    #indxens = np.random.choice(nanals2,size=nanals)
    #xprime = np.dot(enswts.T,xprime2)[indxens]
    return xmean, xprime

def letkf(xmean,xprime,h,obs,oberrvar,obcovlocal):
    """LETKF (with observation localization)"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    # forward operator.
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal])
    hxmean = np.dot(h,xmean)
    obcovlocal = np.where(obcovlocal < 1.e-13,1.e-13,obcovlocal)
    xprime_prior = xprime.copy(); xmean_prior = xmean.copy()
    # brute force application of ETKF for each state element - very slow!
    # assumes all state variables observed.
    ominusf = obs - np.dot(h,xmean_prior)
    for n in range(ndim):
        Rinv = np.diag(obcovlocal[n,:]/oberrvar)
        YbRinv = np.dot(hxprime,Rinv)
        pa = (nanals-1)*np.eye(nanals)+np.dot(YbRinv,hxprime.T)
        evals, eigs = np.linalg.eigh(pa)
        painv = np.dot(np.dot(eigs,np.diag(np.sqrt(1./evals))),eigs.T)
        kfgain = np.dot(xprime_prior[:,n].T,np.dot(np.dot(painv,painv.T),YbRinv))
        enswts = np.sqrt(nanals-1)*painv
        xmean[n] = xmean[n] + np.dot(kfgain, ominusf)
        xprime[:,n] = np.dot(enswts.T,xprime_prior[:,n])
    return xmean, xprime
