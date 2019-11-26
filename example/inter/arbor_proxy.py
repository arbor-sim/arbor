#! /usr/bin/python3

# arbor_proxy.py simulates a arbor run including MPI. Reimplementation of the c++ version of Peyser
# implemented as python to allow easy testing by external developers without building arbor
from mpi4py import MPI
import sys
import numpy as np
import math

#####################################################################
# MPI configuration
world = MPI.COMM_WORLD
rank = world.Get_rank()

# Need to formalize this nicer, currently the coupling is hardcoded
comm = MPI.COMM_WORLD.Split(1) 


########################################################################
# Config
num_arbor_cells = 100;
min_delay = 10;
duration = 100;

arbor_root = 0
nest_root = 1

############################################################################
# Some helper functions

# for debug printing in MPI environment
print_debug = True
print_prefix = "ARB_PROXY_PY: "
def print_d (to_print , force = False):
    if (not (print_debug or force) ):     # print_debug is 'global variable'
        return

    print (print_prefix + str(to_print))
    sys.stdout.flush() # we are debuggin MPI code, force a print after each print statement

def print_spike_array_d (to_print , force = False):
    if (not (print_debug or force) ):     # print_debug is 'global variable'
        return

    # Assume that we received a spike array, no error checking
    print (print_prefix + "SPIKES: [", end = '')
    for spike in to_print:
        print ("S[ " + str(spike[0]) + ":" + str(spike[1]) + " t " + str(spike[2]) + " ]", end = '')
    print ("]")
    sys.stdout.flush() # we are debuggin MPI code, force a print after each print statement

# Gather function
def gather_spikes(spikes, comm):
    # We need to know how much data we will receive in this gather action
    size = comm.size                                 #
    receive_count_array = np.zeros(size, dtype='uint32')
    send_count_array = np.array([spikes.size], dtype='uint32')
    comm.Allgather(send_count_array, receive_count_array)

    # Calculate the amount of spikes 
    cummulative_sum_spikes = np.cumsum(receive_count_array)
    offsets = np.zeros(size) 
    offsets[1:]=cummulative_sum_spikes[:-1] # start with a zero and skip the last entry in cumsum

    # Create buffers for sending and receiving
    # Total nr spikes received is the last entry in cumsum
    # Allgatherv only available as raw byte buffers
    receive_spikes_array = np.ones(cummulative_sum_spikes[-1], dtype='byte')  
    send_buffer = spikes.view(dtype=np.byte)  # send as a byte view in spikes
    receive_buffer = [receive_spikes_array, receive_count_array, offsets, MPI.BYTE]

    comm.Allgatherv(send_buffer, receive_buffer)
    print_spike_array_d(receive_spikes_array.view('uint32,uint32, float32'))

    return receive_spikes_array.view('uint32,uint32, float32')
    

########################################################################
# handshake #1: communicate the number of cells between arbor and nest
# send nr of arbor cells
data_array = np.array([num_arbor_cells],dtype=np.int32)  
world.Bcast(data_array, arbor_root) 

#Receive nest cell_nr
data_array = np.array([0],dtype=np.int32)  
world.Bcast(data_array, root=nest_root) #Use Bcast allows setting of received size
num_nest_cells = data_array[0] 
num_total_cells = num_nest_cells + num_arbor_cells

print_d("num_arbor_cells: " + str(num_arbor_cells) + " " +
        "num_nest_cells: " + str(num_nest_cells) + " " +
        "num_total_cells: " + str(num_total_cells))

########################################################################
# hand shake #2: min delay

# first send the arbor delays
arb_com_time = min_delay / 2.0
data_array = np.array([arb_com_time],dtype=np.float32)  
world.Bcast(data_array, arbor_root)

# receive the nest delays 
data_array = np.array([0],dtype=np.float32)  
world.Bcast(data_array, nest_root)
nest_com_time = data_array[0]
print_d("nest_com_time: " + str(nest_com_time))

###############################################################
# Process the delay and calculate new simulator settings
# TODO: there is some magix going on here, found out why this doubling is done!
double_min_delay = 2 * min(arb_com_time, nest_com_time)
print_d("min_delay: " + str(double_min_delay))
delta = double_min_delay / 2.0
steps = int(math.floor(duration / delta))

# Extra step at end if not a whole multiple
if (steps * delta < duration):
   steps += 1

###############################################################
# Handshake #3: steps
data_array = np.array([steps], dtype=np.int32)  
world.Bcast(data_array, arbor_root)

print_d("delta: " + str(delta) + ", " +
        "sim_duration: " + str(duration) + ", " +
        "steps: " + str(steps) + ", ")

#######################################################
# main simulated simulation loop inclusive nr of steps.
for step in range(steps+1):
    print_d("step: " + str(step) + ": " + str(step * delta))

    # We are sending no spikes from arbor to nest. Create a array with size zero with correct type
    data_array = np.zeros(0, dtype='uint32, uint32, float32')  
    gather_spikes(data_array, world)

print_d("Reached arbor_proxy.py end")




