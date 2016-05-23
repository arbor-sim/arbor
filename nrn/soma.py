from timeit import default_timer as timer
import os.path
from matplotlib import pyplot
import numpy as np
import json
import argparse
from neuron import gui, h

parser = argparse.ArgumentParser(description='generate spike train info for a soma with hh channels')
parser.add_argument('--plot', action='store_true', dest='plot')
args = parser.parse_args()

if args.plot :
    print '-- plotting turned on'
else :
    print '-- plotting turned off'

soma = h.Section(name='soma')

soma.L = soma.diam = 18.8
soma.Ra = 123

print "Surface area of soma =", h.area(0.5, sec=soma)

soma.insert('hh')

stim = h.IClamp(soma(0.5))
stim.delay = 10
stim.dur = 100
stim.amp = 0.1

spike_counter = h.APCount(soma(0.5))
spike_counter.thresh = 0

v_vec = h.Vector()        # Membrane potential vector
t_vec = h.Vector()        # Time stamp vector
s_vec = h.Vector()        # Time stamp vector
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)
spike_counter.record(s_vec)

simdur = 120

# initialize plot
if args.plot :
    pyplot.figure(figsize=(8,4)) # Default figsize is (8,6)

h.tstop = simdur

# run neuron with multiple dt
start = timer()
results = []
for dt in [0.02, 0.01, 0.001, 0.0001]:
    h.dt = dt
    h.run()
    results.append({"dt": dt, "spikes": s_vec.to_python()})
    if args.plot :
        pyplot.plot(t_vec, v_vec, label='neuron ' + str(dt))
end = timer()

print "took ", end-start, " seconds"

# save the spike info as in json format
fp = open('soma.json', 'w')
json.dump(results, fp, indent=1)

if args.plot :
    pyplot.xlabel('time (ms)')
    pyplot.ylabel('mV')
    pyplot.xlim([0, 120])
    pyplot.grid()
    pyplot.legend()
    pyplot.show()

