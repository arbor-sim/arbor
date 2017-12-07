Inhomogeneous Poisson Spike Source
============

A cell that spikes with a variable Poisson distribution. The spike rates can be supplied as a list of times and rates. With optionally linear interpolation between the supplied sample points.

Configuration options:

- start_time    : Time from start when the cell should start spiking (ms.).
- stop_time     : When should the cell stop spiking (ms.)
- sample_delta  : Every sample_delta a test will be made if a spike should be emitted
- rates_per_time: A vector of time-rate pairs. defining the time varying spike rate.
- interpolate   : Should the values be interpolated between the rates supplied in the rates_per_time vector

An mini application is available the generates spikes using this cell source. 
The default time varying inhomogeneous spike rate produced
 
.. code-block:: cpp 
    hz.|                     
    240|     _-_             
       |    -   -  -         
       |   -     -- -        
    0  |__-__________-__     
        100        900   ms 


Notes: 
- The time range is inclusive at the start and exclusive at the stop_time [start, stop)
- The sample_delta steps are independent of the simulation time step.


