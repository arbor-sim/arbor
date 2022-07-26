# Plasticity Example

A miniapp that demonstrates how to use mutable connectivity. The simulation is
run in two parts with different connectivity.

All cells will print out their _generated_ spikes and (cable cells only!) their
membrane potential at the soma. The topology is this
```
         +--------------+
         | 0 Spike      |
         |   Source     |
         | f=1/0.0125ms |
         +--------------+
        /                \ 
       /                  \    <--- This connection will not be present for the first part
      /                    \        and enabled for the second part of the simulation.
     /                      \
     +---------+             +---------+
     | 1 Cable |             | 2 Cable |
     |   Cell  |             |   Cell  |
     +---------+             +---------+
```
