Regular Spike Source
============

A simple cell that spikes with a regular rate for a set period. 

Configuration options:
- start_time: Time from start when the cell should start spiking (ms.).
- period    : Every period a spike will be generated (ms.).
- stop_time : When should the cell stop spiking (ms.)

Notes: 
- The time range is inclusive at the start and exclusive at the stop_time [start, stop)
- The generates spike times are independent of the simulation time step.
