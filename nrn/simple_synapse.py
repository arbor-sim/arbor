from timeit import default_timer as timer
import os.path
from matplotlib import pyplot
import numpy as np
import json
import argparse
from neuron import gui, h

parser = argparse.ArgumentParser(description='generate spike train ball and stick model with hh channels at soma and pas channels in dendrite')
parser.add_argument('--plot', action='store_true', dest='plot')
args = parser.parse_args()

soma = h.Section(name='soma')
dend = h.Section(name='dend')

dend.connect(soma(1))

# Surface area of cylinder is 2*pi*r*h (sealed ends are implicit).
# Here we make a square cylinder in that the diameter
# is equal to the height, so diam = h. ==> Area = 4*pi*r^2
# We want a soma of 500 microns squared:
# r^2 = 500/(4*pi) ==> r = 6.2078, diam = 12.6157
soma.L = soma.diam = 12.6157 # Makes a soma of 500 microns squared.
dend.L = 200 # microns
dend.diam = 1 # microns

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

# create a stimulator that stimulates once at 9 ms
stim = h.NetStim()
stim.number = 1
stim.start  = 9

if args.plot :
    pyplot.figure(figsize=(8,4)) # Default figsize is (8,6)
    pyplot.grid()

simdur = 50.0
h.tstop = simdur
h.dt = 0.01

start = timer()
results = []
for nseg in [5, 11, 51, 101] :

    print 'simulation with ', nseg, ' compartments in dendrite...'

    dend.nseg=nseg

    # add a synapse
    syn_ = h.ExpSyn(dend(0.5))
    syn_.tau = 2
    ncstim = h.NetCon(stim, syn_)
    ncstim.delay = 1 # 1 ms delay (arrives at 10ms)
    ncstim.weight[0] = 0.04 # NetCon weight is a vector

    ncstim1 = h.NetCon(stim, syn_)
    ncstim1.delay = 11 # 1 ms delay (arrives at 10ms)
    ncstim1.weight[0] = 0.04 # NetCon weight is a vector
    ncstim2 = h.NetCon(stim, syn_)
    ncstim2.delay = 31 # 1 ms delay (arrives at 10ms)
    ncstim2.weight[0] = 0.04 # NetCon weight is a vector

    # record voltages
    v_soma = h.Vector() # soma
    v_dend = h.Vector() # middle of dendrite

    v_soma.record( soma(0.5)._ref_v)
    v_dend.record( dend(0.5)._ref_v)

    # record spikes
    spike_counter_soma = h.APCount(soma(0.5))
    spike_counter_soma.thresh = 20
    spike_counter_dend = h.APCount(dend(0.5))
    spike_counter_dend.thresh = -10

    spikes_soma = h.Vector() # soma
    spikes_dend = h.Vector() # middle of dendrite

    spike_counter_soma.record(spikes_soma)
    spike_counter_dend.record(spikes_dend)

    # record time stamps
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    # finally it's time to run the simulation
    h.run()

    results.append(
        {
            "nseg": nseg,
            "dt" : h.dt,
            "measurements": {
               "soma" : {
                   "thresh" :  spike_counter_soma.thresh,
                   "spikes" :  spikes_soma.to_python()
               },
               "dend" : {
                   "thresh" :  spike_counter_dend.thresh,
                   "spikes" :  spikes_dend.to_python()
               },
           }
        }
    )

    if args.plot :
        pyplot.plot(t_vec, v_soma,  'k', linewidth=1, label='soma ' + str(nseg))
        pyplot.plot(t_vec, v_dend,  'b', linewidth=1, label='dend ' + str(nseg))

# time the simulations
end = timer()
print "took ", end-start, " seconds"

# save the spike info as in json format
fp = open('simple_synapse.json', 'w')
json.dump(results, fp, indent=2)

if args.plot :
    pyplot.xlabel('time (ms)')
    pyplot.ylabel('mV')
    pyplot.xlim([0, simdur])
    pyplot.legend()

    pyplot.show()

