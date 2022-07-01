# Plasticity Example

A miniapp that demonstrates how to use mutable connectivity. Should be run with
MPI and to see something interesting at least two ranks should be used.

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
There is one cell per rank s.t. `gid == rank` and `gid 0` is reserved for the spike source.

