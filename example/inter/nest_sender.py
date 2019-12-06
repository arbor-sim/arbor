#! /usr/bin/python3

# This is the real nest program, which requires NESTIO + ARBOR-NESTIO

from sys import argv
argv.append('--quiet')
import sys

print("Getting comm")
from mpi4py import MPI
comm = MPI.COMM_WORLD.Split(0) # is nest

print("Getting nest")
import nest
nest.set_communicator(comm)
nest.SetKernelStatus({'recording_backends': {'arbor':{}}})

print("Building network")

# Create a spike generator
pg = nest.Create('poisson_generator', params={'rate': 10.0})

# Poisson_generators are special, we need a parrot to forward the spikes to be able
# to record
g = nest.Create('poisson_generator', params={'rate': 10.0})
nest.Connect(pg, parrots)

# can now record from the parrots.
sd2 = nest.Create('spike_detector', params={"record_to": "arbor"})			  
nest.Connect(parrots, sd2)

status = nest.GetKernelStatus()
print('min_delay: ', status['min_delay'], ", max_delay: ", status['max_delay'])

print("Simulate")
sys.stdout.flush()

nest.Simulate(100.0)

print("Done")
sys.stdout.flush()