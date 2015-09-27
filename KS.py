import numpy as np

class KS(object):
    #
    # Solution of 1-d Kuramoto-Sivashinsky equation, the simplest
    # PDE that exhibits spatio-temporal chaos.
    # refs:
    # http://people.maths.ox.ac.uk/trefethen/pdectb/kuramoto2.pdf
    # http://www.encyclopediaofmath.org/index.php/Kuramoto-Sivashinsky_equation
    # http://onlinelibrary.wiley.com/doi/10.1002/fld.2020/pdf
    # http://sprott.physics.wisc.edu/pubs/paper335.pdf
    #
    # u_t = -u*u_x - u_xx - diffusion*u_xxxx, periodic BCs on [0,2*pi*L].
    # time step dt with N fourier collocation points.
    # energy enters the system at long wavelengths via u_xx,
    # (an unstable diffusion term),
    # cascades to short wavelengths due to the nonlinearity u*u_x, and
    # dissipates via diffusion*u_xxxx.
    # Order of diffusion can be varied to control character of power spectrum.
    # (exponent=4 is uxxxx diffusion, exponent=8 or 16 also work, but
    # diffusion coefficient must be reduced to maintain numerical stability)
    #
    def __init__(self,L=16,N=128,dt=1.0,exponent=4,diffusion=1.0,members=1):
        self.L = L
        self.n = N
        self.members = members
        self.dt = dt
        self.diffusion = diffusion
        self.exponent = exponent
        kk = N*np.fft.fftfreq(N)[0:(N/2)+1]  # wave numbers
        self.wavenums = kk
        k  = kk.astype(np.float)/L
        self.ik    = 1j*k              # spectral derivative operator
        self.lin   = k**2 - diffusion*(k**exponent)  # Fourier multipliers for linear term
        # random noise initial condition.
        u = 0.01*np.random.standard_normal(size=(members,N))
        # remove zonal mean from initial condition.
        u = u - u.mean()
        # spectral space variable
        self.xspec = np.fft.rfft(u)
        self.x = u
    def nlterm(self,v):
        # compute tendency from nonlinear term.
        u = np.fft.irfft(v,axis=-1)
        return -0.5*self.ik*np.fft.rfft(u**2,axis=-1)
    def advance(self):
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        self.xspec = np.fft.rfft(self.x,axis=-1)
        vsave = self.xspec.copy()
        v = self.xspec.copy()
        for n in range(3):
            dt = self.dt/(3-n)
            # explicit RK3 step for nonlinear term
            v = vsave + dt*self.nlterm(v)
            # implicit trapezoidal adjustment for linear term
            v = (v+0.5*self.lin*dt*vsave)/(1.-0.5*self.lin*dt)
        self.xspec = v
        self.x = np.fft.irfft(v,axis=-1)
        # Lorenz 3-cycle + trapezoidal
        # ref: http://dx.doi.org/10.1175/MWR-D-13-00132.1

        #a21 = 1./3.
        #a31 = 1./6.; a32 = 1./2. # scheme 1
        #b1 = 0.5; b2 = -0.5; b3 = 1. # scheme 1
        ##a31 = -1./3.; a32 = 1. # scheme 2
        ##b1 = 0.; b2 = 0.5; b3 = 0.5 # scheme 2
        #aa21 = 1./6.; aa22 = 1./6.
        #aa31 = 1./3;  aa32 = 0.; aa33 = 1./3.
        #b = a31+a32
        #c = 1./5.
        #d = (1./b)*(0.5-c)
        #bb1 = 1-(c+d); bb2 = 0.; bb3 = d; bb4 = c

        ## stage 1
        #nlterm1 = dt*self.nlterm(v1)
        #linterm = dt*self.lin
        #v2 = v1 + a21*nlterm1 + aa21*v1*linterm
        #v2 = v2/(1.-aa22*linterm)
        ## stage 2
        #nlterm2 = dt*self.nlterm(v2)
        #v3 = v1 + a31*nlterm1 + a32*nlterm2 +\
        #          aa31*v1*linterm + aa32*v2*linterm
        #v3 = v3/(1.-aa33*linterm)
        ## stage 3
        #nlterm3 = dt*self.nlterm(v3)
        #v = v1 + nlterm1*b1 + nlterm2*b2 + nlterm3*b3 +\
        #         bb1*linterm*v1 + bb2*linterm*v2 + bb3*linterm*v3
        #v = v/(1.-bb4*linterm)
