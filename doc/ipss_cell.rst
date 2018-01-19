Inhomogeneous Poisson Spike Source
============

A cell that spikes with a variable Poisson distribution. The spike rates can be supplied as a list of times and rates. With optionally linear interpolation between the supplied sample points.

Spikes are generated at a configurable sample rate. With a rate that can be
varied across time. 
All times supplied to the cell will be 'rounded' to the first higher multiple of
of the supplied sample_delta.
When interpolating the rate will be the averaged rate between the
t and t+1. The return spike time will be t.

The first rate supplied in the time-rate vector should be at or before the
start time of the cell. The last supplied time-rate will be kept
until the stop_time of the cell

Configuration is done with the class arb::ipss_cell_description

+---------------+---------------+-----------------------------------------------------------+
| Parameter     |  type         |  Details                                                  |
+===============+===============+===========================================================+
| start_time    |  float        |  Time from start when the cell should start spiking (ms.).|
+---------------+---------------+-----------------------------------------------------------+
| stop_time     |  float        |  When should the cell stop spiking (ms.)                  |
+---------------+---------------+-----------------------------------------------------------+
| sample_delta  |  float        |  Every sample_delta entry a dice will be rolled of a      |
|               |               |  spike should be emitted                                  |
+---------------+---------------+-----------------------------------------------------------+
| rates_per_time|  vector       |  A vector of time-rate pairs defining the time varying    |
|               |  <float,double>|  spike rate.                                              |
+---------------+---------------+-----------------------------------------------------------+
| interpolate   |  bool         |  Should the values be interpolated between the rates      |
|               |               |  supplied in the rates_per_time vector                    |
+---------------+---------------+-----------------------------------------------------------+

An mini application is available to illustrate using this cell source: 

- /miniapps/ipss/ipss.exe
- /miniapps/ipss/parse_and_plot.py

.. image:: https://i.imgur.com/bprO9Ek.png
    :alt: inhomogeneous spike 

Default output of the ipss app. Average spike rate of 10000 IPSS cells time binned with a 1 ms. bin. 
The green line is the target frequency as supplied on the configuration.    
    
Notes: 

- The start time range is inclusive while the stop_time is exclusive [start, stop)
- The sample_delta steps are independent of the simulation time step.
