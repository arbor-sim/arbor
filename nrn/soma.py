import os.path
from neuron import h, gui
from matplotlib import pyplot
import numpy as np

soma = h.Section(name='soma')

soma.L = soma.diam = 18.8
soma.Ra = 123

print "Surface area of soma =", h.area(0.5, sec=soma)

soma.insert('hh')

stim = h.IClamp(soma(0.5))
stim.delay = 10
stim.dur = 100
stim.amp = 0.1

v_vec = h.Vector()        # Membrane potential vector
t_vec = h.Vector()        # Time stamp vector
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)
simdur = 120

# initialize plot
pyplot.figure(figsize=(8,4)) # Default figsize is (8,6)

pyplot.subplot(2,1,1)
pyplot.grid()

h.tstop = simdur

# run neuron with multiple dt

for dt in [0.02, 0.01, 0.005, 0.002, 0.001]:
    h.dt = dt
    h.run()
    pyplot.plot(t_vec, v_vec, label='neuron ' + str(dt))
    #pyplot.plot(t_vec, v_vec, 'k', label='neuron ' + str(dt))

pyplot.xlim([102.5,105])
pyplot.ylim([0,40])

pyplot.legend()
pyplot.subplot(2,1,2)
pyplot.grid()
step = 0
while os.path.isfile('../tests/v_' + str(step) + '.dat') :
    fname = '../tests/v_' + str(step) + '.dat'
    print 'loading ' + fname
    data = np.loadtxt(fname)
    t = data[:,0]
    v = data[:,1]
    #pyplot.plot(t, v, 'b', label=fname)
    pyplot.plot(t, v, label=fname)
    step = step + 1

pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')

pyplot.xlim([102.5,105])
pyplot.ylim([0,40])

pyplot.legend()
pyplot.show()

