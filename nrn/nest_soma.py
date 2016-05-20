import os.path
from matplotlib import pyplot
import numpy as np

step = 0
while os.path.isfile('../tests/v_' + str(step) + '.dat') :
    fname = '../tests/v_' + str(step) + '.dat'
    print 'loading ' + fname
    data = np.loadtxt(fname)
    t = data[:,0]
    v = data[:,1]
    pyplot.plot(t, v, 'b', label=fname)
    step = step + 1

pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')

pyplot.xlim([87.5,89.5])
#pyplot.ylim([0,40])

pyplot.legend()
pyplot.show()

