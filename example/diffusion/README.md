# Diffusion example.

Example of simulating a simple linear neuron with a diffusing sodium
concentration. An event injecting more Na into the centre is fired at t=0. The
measured concentrations will be written to disk further analysis.

 |    |  Option  | Meaning                         | Default     |
 |----|----------|---------------------------------|-------------|
 | -t | --tfinal | Length of the simulation period | 1 ms        |
 | -d | --dt     | Simulation time step            | 0.01 ms     |
 | -s | --ds     | Sampling interval               | 0.1 ms      |
 | -g | --gpu    | Use GPU id, enabled if >=0      | -1          |
 | -l | --length | Length of stick                 | 30 um       |
 | -x | --dx     | Discretisation                  | 1 um        |
 | -i | --Xi     | Initial Na concentration        | 0 mM        |
 | -b | --beta   | Na diffusivity                  | 0.005 m^2/s |
 | -o | --output | Save samples                    | log.csv     |
 
