from matplotlib import pyplot
import numpy as np

data = np.loadtxt('v.dat')
t = data[:,0]
nfields = data.shape[1]

for i in range(1,nfields) :
    v = data[:,i]
    pyplot.plot(t, v)

pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')
pyplot.grid()
pyplot.show()


#mi = min(v)
#ma = max(v)
#rng = (ma-mi)/2 * 1.1
#mn = 0.5*(mi+ma)
#pyplot.ylim(mn-rng, mn+rng)
