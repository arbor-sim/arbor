from neuron import h, gui
import numpy as np

soma = h.Section(name='soma')
dend = h.Section(name='dend')

dend.connect(soma(1))

h.psection(sec=soma)

# Surface area of cylinder is 2*pi*r*h (sealed ends are implicit).
# Here we make a square cylinder in that the diameter
# is equal to the height, so diam = h. ==> Area = 4*pi*r^2
# We want a soma of 500 microns squared:
# r^2 = 500/(4*pi) ==> r = 6.2078, diam = 12.6157
soma.L = soma.diam = 12.6157 # Makes a soma of 500 microns squared.
dend.L = 200 # microns
dend.diam = 1 # microns
print "Surface area of soma =", h.area(0.5, sec=soma)

for sec in h.allsec():
    sec.Ra = 100    # Axial resistance in Ohm * cm
    sec.cm = 1      # Membrane capacitance in micro Farads / cm^2

# Insert active Hodgkin-Huxley current in the soma
soma.insert('hh')
soma.gnabar_hh = 0.12  # Sodium conductance in S/cm2
soma.gkbar_hh = 0.036  # Potassium conductance in S/cm2
soma.gl_hh = 0.0003    # Leak conductance in S/cm2
soma.el_hh = -54.3     # Reversal potential in mV

# Insert passive current in the dendrite
dend.insert('pas')
dend.g_pas = 0.001  # Passive conductance in S/cm2
dend.e_pas = -65    # Leak reversal potential mV

#for sec in h.allsec():
#    h.psection(sec=sec)

stim = h.IClamp(dend(1))
stim.delay = 5
stim.dur = 1
stim.amp = 5*0.1

somastyles = ['b-', 'r-']
dendstyles = ['b-.', 'r-.']
dt = [0.02, 0.001]
nsegs = [10, 100]
from matplotlib import pyplot
pyplot.figure(figsize=(8,4)) # Default figsize is (8,6)
pyplot.grid()
pos = 0
for pos in [0] :
    dend.nseg=nsegs[pos]

    v_soma = h.Vector()        # Membrane potential vector
    v_dend = h.Vector()        # Membrane potential vector
    t_vec = h.Vector()        # Time stamp vector
    v_soma.record(soma(0.5)._ref_v)
    v_dend.record(dend(1.0)._ref_v)
    t_vec.record(h._ref_t)
    simdur = 25.0

    h.tstop = simdur
    h.dt = dt[pos]
    h.run()

    pyplot.plot(t_vec, v_soma, somastyles[pos], linewidth=2)
    pyplot.plot(t_vec, v_dend, dendstyles[pos], linewidth=2)

pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')

data = np.loadtxt('../tests/v.dat')
t = data[:,0]
nfields = data.shape[1]

v = data[:,1]
pyplot.plot(t, v, 'g', linewidth=2)
v = data[:,2]
pyplot.plot(t, v, 'g-.', linewidth=2)

pyplot.show()

