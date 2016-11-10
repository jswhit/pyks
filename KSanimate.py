import numpy as np
from KSvaryingdiff import KS
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L   = 32           # domain is 0 to 2.*np.pi*L
N   = 512          # number of collocation points
dt  = 0.3          # time step
diffusion = 1.0
ks = KS(L=L,diffusion=diffusion,N=N,dt=dt) # instantiate model

# define initial condition
#u = np.cos(x/L)*(1.0+np.sin(x/L)) # smooth IC
u = 0.01*np.random.normal(size=N) # noisy IC
# remove zonal mean
u = u - u.mean()
# spectral space variable.
ks.xspec[0] = np.fft.rfft(u)

# time stepping loop.
nmin = 400; nmax = 2000
uu = []; tt = []
vspec = np.zeros(ks.xspec.shape[1], np.float)
x = np.arange(N)
fig, ax = plt.subplots()
line, = ax.plot(x, ks.x.squeeze())
ax.set_xlim(0,N-1)
ax.set_ylim(-10,10)
#Init only required for blitting to give a clean slate.
def init():
    global line
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

# spinup
for n in range(nmin):
    ks.advance()

def updatefig(n):
    global tt,uu,vspec
    ks.advance()
    vspec += np.abs(ks.xspec.squeeze())**2
    u = ks.x.squeeze()
    line.set_ydata(u)
    print n,u.min(),u.max()
    uu.append(u); tt.append(n*dt)
    return line,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, updatefig, np.arange(1,nmax+1), init_func=init,
                              interval=25, blit=True, repeat=False)
ani.save('KS.mp4',writer=writer)
plt.show()

plt.figure()
# make contour plot of solution, plot spectrum.
ncount = len(uu)
vspec = vspec/ncount
uu = np.array(uu); tt = np.array(tt)
print tt.min(), tt.max()
print uu.shape
uup = uu - uu.mean(axis=0)
print uup.shape
cov = np.dot(uup.T,uup)
print 'cov',cov.min(), cov.max(), cov.shape
nplt = 800
plt.contourf(x,tt[:nplt],uu[:nplt],31,cmap=plt.cm.spectral,extend='both')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.title('chaotic solution of the K-S equation')


plt.figure()
plt.loglog(ks.wavenums,vspec)
plt.title('variance spectrum')
#plt.ylim(0.001,10000)
#plt.xlim(0,100)

plt.figure()
plt.contourf(x,x,cov,31,cmap=plt.cm.spectral,extend='both')

plt.show()
