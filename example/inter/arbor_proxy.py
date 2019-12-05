#! /usr/bin/python3

# arbor_proxy.py simulates a arbor run including MPI. Reimplementation of the c++ version of Peyser
# implemented as python to allow easy testing by external developers without building arbor
from mpi4py import MPI
import sys
import numpy as np
import math

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
    sys.stdout.flush() 


# we are debuggin MPI code, force a print after each print statement
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
    
class comm_information(): 
    def __init__(self, is_arbor):
        self.is_arbor = is_arbor
        self.is_nest = not is_arbor

        self.global_rank = MPI.COMM_WORLD.rank
        self.global_size = MPI.COMM_WORLD.size

        # split MPI_COMM_WORLD: all arbor go into split 1
        # TODO: with N>2 simulators self whole function needs to be cleaned up
        color = 1 if is_arbor else 0
        self.world = MPI.COMM_WORLD
        self.comm = self.world.Split(color)

        local_size  = self.comm.size
        self.local_rank = self.comm.rank;

        self.arbor_size = local_size if self.is_arbor else self.global_size - local_size
        self.nest_size = self.global_size - self.arbor_size

        input = np.array([self.global_rank], dtype=np.int32) 
        local_ranks = np.zeros(local_size, dtype=np.int32)
       
        self.comm.Allgather(input, local_ranks)
        local_ranks.sort()

        # look for first non concecutive entry. self would occur if we create the ranks interleaved
        def first_missing(np_array):
            for idx in range(np_array.size-1):
                
                if not (np_array[idx+1] - np_array[idx] is 1):
                    return np_array[idx] + 1
            # Default the last rank plus one
            return np_array[-1]+1

        if (self.is_arbor):
            self.arbor_root = local_ranks[0]
            print ("self.arbor_root:" + str(self.arbor_root))
            print ("first_missing(local_ranks) " + str(first_missing(local_ranks)) )

            self.nest_root = first_missing(local_ranks) if self.arbor_root == 0 else 0
        else: 
            self.nest_root = local_ranks[0]
            self.arbor_root = first_missing(local_ranks) if self.nest_root == 0 else 0


        print_d("self.nest_root" +str(self.nest_root))

    def __str__(self):
        return str("global ( rank: " + str(self.global_rank) + ", size: " + str(self.global_size) + "\n" +
             "local rank " + str(self.local_rank) + "\n" +
             "self is arbor\n" if self.is_arbor else "self is nest\n" +
             "arbor (root: " + str(self.arbor_root) + ", size: " + str(self.arbor_size) + ")\n" +
             +"nest (root: " + str(self.nest_root) + ", size: " + str(self.nest_size) + ")\n")

#####################################################################
# MPI configuration
comm_info = comm_information(True)

########################################################################
# Config
num_arbor_cells = 100;
min_delay = 10;
duration = 100;

arbor_root = 0
nest_root = 1


########################################################################
# handshake #1: communicate the number of cells between arbor and nest
# send nr of arbor cells
data_array = np.array([num_arbor_cells],dtype=np.int32)  
comm_info.world.Bcast(data_array, comm_info.arbor_root) 

#Receive nest cell_nr
data_array = np.array([0],dtype=np.int32)  
comm_info.world.Bcast(data_array, root=comm_info.nest_root) #Use Bcast allows setting of received size

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
comm_info.world.Bcast(data_array, comm_info.arbor_root)

# receive the nest delays 
data_array = np.array([0],dtype=np.float32)  
comm_info.world.Bcast(data_array, comm_info.nest_root)
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
comm_info.world.Bcast(data_array, comm_info.arbor_root)

print_d("delta: " + str(delta) + ", " +
        "sim_duration: " + str(duration) + ", " +
        "steps: " + str(steps) + ", ")

#######################################################
# main simulated simulation loop inclusive nr of steps.
for step in range(steps+1):
    print_d("step: " + str(step) + ": " + str(step * delta))

    # We are sending no spikes from arbor to nest. Create a array with size zero with correct type
    data_array = np.zeros(0, dtype='uint32, uint32, float32')  
    gather_spikes(data_array, comm_info.world)

print_d("Reached arbor_proxy.py end")




