from matplotlib import pyplot
import numpy as np

data = np.loadtxt('v.dat')
t = data[:,0]
v = data[:,1]


pyplot.plot(t, v)
pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')
pyplot.show()

