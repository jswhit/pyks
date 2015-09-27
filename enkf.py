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
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    #for nanal in range(nanals):
    #    hxprime[nanal] = np.dot(h,xprime[nanal])
    #hxmean = np.dot(h,xmean)
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(ndim),xmean)
    #plt.plot(np.arange(ndim),hxmean)
    #plt.show()
    #raise SystemExit
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        for nanal in range(nanals):
            hxprime[nanal] = np.dot(h,xprime[nanal])
        hxmean = np.dot(h,xmean)
        # state space update
        hxens = hxprime[:,nob].reshape((nanals, 1))
        hpbht = (hxens**2).sum()/(nanals-1)
        gainfact = ((hpbht+oberrvar)/hpbht*\
                   (1.-np.sqrt(oberrvar/(hpbht+oberrvar))))
        pbht = (xprime.T*hxens[:,0]).sum(axis=1)/float(nanals-1)
        kfgain = covlocal[nob,:]*pbht/(hpbht+oberrvar)
        xmean = xmean + kfgain*(ob-hxmean[nob])
        xprime = xprime - gainfact*kfgain*hxens
    return xmean, xprime

def serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z):
    """serial potter method"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]

    if z is None:
        # set ensemble to square root of localized Pb
        Pb = covlocal*np.dot(xprime.T,xprime)/(nanals-1)
        evals, eigs = np.linalg.eigh(Pb)
        print evals.sum()
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
                nanal2 += 1
        xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2
    #print xprime2.shape
    #print ((xprime**2).sum(axis=0)/(nanals-1)).sum()
    #print ((xprime2**2).sum(axis=0)/(nanals2-1)).sum()
    #raise SystemExit

    # update xmean using full xprime2
    # update original xprime using gain from full xprime2
    hxprime = np.empty((nanals2, nobs), xprime.dtype)
    hxprime_orig = np.empty((nanals, nobs), xprime.dtype)
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        for nanal in range(nanals2):
            hxprime[nanal] = np.dot(h,xprime2[nanal])
        for nanal in range(nanals):
            hxprime_orig[nanal] = np.dot(h,xprime[nanal])
        hxmean = np.dot(h,xmean)
        # state space update
        hxens = hxprime[:,nob].reshape((nanals2, 1))
        hxens_orig = hxprime_orig[:,nob].reshape((nanals, 1))
        hpbht = (hxens**2).sum()/(nanals2-1)
        gainfact = ((hpbht+oberrvar)/hpbht*\
                   (1.-np.sqrt(oberrvar/(hpbht+oberrvar))))
        pbht = (xprime2.T*hxens[:,0]).sum(axis=1)/float(nanals2-1)
        kfgain = pbht/(hpbht+oberrvar)
        xmean = xmean + kfgain*(ob-hxmean[nob])
        xprime2 = xprime2 - gainfact*kfgain*hxens
        xprime  = xprime  - gainfact*kfgain*hxens_orig
    #print ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    #print ((xprime**2).sum(axis=0)/(nanals-1)).mean()
    #raise SystemExit
    # update original perts with (unmodulated) serial ensrf.
    #hxprime = np.empty((nanals, nobs), xprime.dtype)
    #for nob,ob in zip(np.arange(nobs),obs):
    #    # forward operator.
    #    for nanal in range(nanals):
    #        hxprime[nanal] = np.dot(h,xprime[nanal])
    #    # state space update
    #    hxens = hxprime[:,nob].reshape((nanals, 1))
    #    hpbht = (hxens**2).sum()/(nanals-1)
    #    gainfact = ((hpbht+oberrvar)/hpbht*\
    #               (1.-np.sqrt(oberrvar/(hpbht+oberrvar))))
    #    pbht = (xprime.T*hxens[:,0]).sum(axis=1)/float(nanals-1)
    #    kfgain = covlocal[nob,:]*pbht/(hpbht+oberrvar)
    #    xprime = xprime - gainfact*kfgain*hxens
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

def etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z):
    """ETKF (use only with full rank ensemble, no localization)"""
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
    # make square root symmetric.
    painv = np.dot(np.dot(eigs,np.diag(np.sqrt(1./evals))),eigs.T)
    kfgain = np.dot(xprime2.T,np.dot(np.dot(painv,painv.T),YbRinv))
    enswts = np.sqrt(nanals2-1)*painv
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    # random sample of posterior modulated ensemble to update ens perts.
    #xprime2 = np.dot(enswts.T,xprime2)
    ##print ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    #nanals_ran = np.random.choice(np.arange(nanals2),nanals)
    #xprime = xprime2[nanals_ran]
    #xprime = xprime - xprime.mean(axis=0) # ensure zero mean
    #print ((xprime**2).sum(axis=0)/(nanals-1)).mean()
    #raise SystemExit
    # update perts with serial ensrf.
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        for nanal in range(nanals):
            hxprime[nanal] = np.dot(h,xprime[nanal])
        # state space update
        hxens = hxprime[:,nob].reshape((nanals, 1))
        hpbht = (hxens**2).sum()/(nanals-1)
        gainfact = ((hpbht+oberrvar)/hpbht*\
                   (1.-np.sqrt(oberrvar/(hpbht+oberrvar))))
        pbht = (xprime.T*hxens[:,0]).sum(axis=1)/float(nanals-1)
        kfgain = covlocal[nob,:]*pbht/(hpbht+oberrvar)
        xprime = xprime - gainfact*kfgain*hxens
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
