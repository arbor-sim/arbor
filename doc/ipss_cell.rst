Inhomogeneous Poisson Spike Source
============

A cell that spikes with a variable Poisson distribution. The spike rates can be supplied as a list of times and rates. With optionally linear interpolation between the supplied sample points.

Configuration options:

+---------------+------+-------------------------------------+
| Parameter     | type | Details                             |
+===============+======+=====================================+
| start_time    | float| Time from start when the cell should start spiking (ms.).|
+---------------+------+-------------------------------------+
| stop_time     | float| When should the cell stop spiking (ms.)|
+---------------+------+-------------------------------------+
| sample_delta  | vector<time,float> | Every sample_delta a test will be made if a spike should be emitted |
+---------------+------+-------------------------------------+
| rates_per_time| A vector of time-rate pairs. defining the time varying spike rate.
+---------------+------+-------------------------------------+
|interpolate    | bool |Should the values be interpolated between the rates supplied in the rates_per_time vector |
+---------------+------+-------------------------------------+
An mini application is available the generates spikes using this cell source: 
miniapps/ipss/ipss.exe

.. image:: https://i.imgur.com/bprO9Ek.png
    :alt: inhomogeneous spike 

Average spike rate of 10000 IPSS cells time binned with a 1 ms. bin. 
The green line is the target frequency as supplied on the configuration.    
    
Notes: 

- The time range is inclusive at the start and exclusive at the stop_time [start, stop)
- The sample_delta steps are independent of the simulation time step.
