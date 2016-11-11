import numpy as np

class KS(object):
    #
    # Solution of 1-d Kuramoto-Sivashinsky equation, the simplest
    # PDE that exhibits spatio-temporal chaos
    # (https://www.encyclopediaofmath.org/index.php/Kuramoto-Sivashinsky_equation).
    #
    # u_t + u*u_x + u_xx + diffusion*u_xxxx = 0, periodic BCs on [0,2*pi*L].
    # time step dt with N fourier collocation points.
    # energy enters the system at long wavelengths via u_xx,
    # (an unstable diffusion term),
    # cascades to short wavelengths due to the nonlinearity u*u_x, and
    # dissipates via diffusion*u_xxxx.
    #
    def __init__(self,L=16,N=128,dt=0.5,diffusion=1.0,members=1,rs=None):
        self.L = L; self.n = N; self.members = members; self.dt = dt
        self.diffusion = diffusion
        kk = N*np.fft.fftfreq(N)[0:(N/2)+1]  # wave numbers
        self.wavenums = kk
        k  = kk.astype(np.float)/L
        self.k     = k
        self.ik    = 1j*k                   # spectral derivative operator
        # random noise initial condition.
        if rs is None:
            rs = np.random.RandomState()
        x = 0.01*rs.standard_normal(size=(members,N))
        diff_fact = 2.0; exponent = 4
        self.blend = np.cos(np.linspace(0,np.pi,N))**exponent
        self.lin   = k**2 - (diffusion/diff_fact)*k**4  # Fourier multipliers for linear term
        self.lin2  = k**2 - diff_fact*diffusion*k**4  # Fourier multipliers for linear term
        # remove zonal mean from initial condition.
        self.x = x - x.mean()
        # spectral space variable
        self.xspec = np.fft.rfft(self.x,axis=-1)
    def nlterm(self,xspec):
        # compute tendency from nonlinear term.
        x = np.fft.irfft(xspec,axis=-1)
        return -0.5*self.ik*np.fft.rfft(x**2,axis=-1)
    def advance(self):
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        self.xspec = np.fft.rfft(self.x,axis=-1)
        xspec_save = self.xspec.copy()
        for n in range(3):
            dt = self.dt/(3-n)
            # explicit RK3 step for nonlinear term
            self.xspec = xspec_save + dt*self.nlterm(self.xspec)
            # implicit trapezoidal adjustment for linear term
            self.xspec = (self.xspec+0.5*self.lin*dt*xspec_save)/(1.-0.5*self.lin*dt)
        self.xspec1 = self.xspec.copy()
        self.xspec = xspec_save.copy()
        for n in range(3):
            dt = self.dt/(3-n)
            # explicit RK3 step for nonlinear term
            self.xspec = xspec_save + dt*self.nlterm(self.xspec)
            # implicit trapezoidal adjustment for linear term
            self.xspec =\
            (self.xspec+0.5*self.lin2*dt*xspec_save)/(1.-0.5*self.lin2*dt)
        x1 = np.fft.irfft(self.xspec1,axis=-1)
        x2 = np.fft.irfft(self.xspec,axis=-1)
        # blend high and low diffusion solutions.
        self.x = (1.-self.blend)*x1 + self.blend*x2
        self.xspec = np.fft.rfft(self.x,axis=-1)
