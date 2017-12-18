from matplotlib import pyplot
import numpy as np

ncol = 3

raw = np.fromfile("cell0.txt", sep=" ")
n = raw.size/ncol
data = raw.reshape(n,ncol)

t    = data[:, 0]
soma = data[:, 1]
dend = data[:, 2]

pyplot.plot(t, soma, 'k')
pyplot.plot(t, dend, 'r')

pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')
pyplot.xlim([t[0], t[n-1]])
pyplot.grid()
pyplot.show()

