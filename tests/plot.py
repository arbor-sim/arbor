from matplotlib import pyplot
import numpy as np

raw = np.fromfile("den_syn.txt", sep=" ")
n = raw.size/3
data = raw.reshape(n,3)

t    = data[:, 0]
soma = data[:, 1]
dend = data[:, 2]

pyplot.plot(t, soma)
pyplot.plot(t, dend)

pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')
pyplot.xlim([t[0], t[n-1]])
pyplot.grid()
pyplot.show()

