# Plasticity Example

A miniapp that demonstrates how to use mutable connectivity. Must be run with
MPI-enabled and **exactly** three ranks.

Each rank will print out its _generated_ spikes. The topology is this
```
         +--------------+
         | 0 Spike      |
         |   Source     |
         | f=1/0.0125ms |
         +--------------+
        /                \ 
       /                  \    <--- This connection will not be present for t=0..0.25ms
      /                    \        and enabled from 0.25..0.5ms
     /                      \
     +---------+             +---------+
     | 1 Cable |             | 2 Cable |
     |   Cell  |             |   Cell  |
     +---------+             +---------+
```
There is one cell per rank s.t. `gid == rank`.

