#! /usr/bin/python

# This is the real nest program, which requires NESTIO + ARBOR-NESTIO

from sys import argv
argv.append('--quiet')

print("Getting comm")
from mpi4py import MPI
comm = MPI.COMM_WORLD.Split(0) # is nest

print("Getting nest")
import nest
nest.set_communicator(comm)

print("Building network")
pg = nest.Create('poisson_generator', params={'rate': 10.0})
parrots = nest.Create('parrot_neuron', 100)
nest.Connect(pg, parrots)
sd = nest.Create('spike_detector',
                 params={"record_to": ["screen", "arbor"]})
nest.Connect(parrots, sd)

#print(nest.GetKernelStatus())
#nest.SetKernelStatus({'recording_backend': 'arbor'})
#nest.SetKernelStatus({'recording_backends': {'screen': {}}})
# status = nest.GetKernelStatus()
# print('min_delay: ', status['min_delay'], ", max_delay: ", status['max_delay'])
# nest.SetKernelStatus({'min_delay': status['min_delay']/2,
#                       'max_delay': status['max_delay']})
# status = nest.GetKernelStatus()
# print('min_delay: ', status['min_delay'], ", max_delay: ", status['max_delay'])

print("Simulate")
nest.Simulate(100.0)
print("Done")
