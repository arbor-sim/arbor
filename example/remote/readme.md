# Remote Communication with External Simulators

This example shows
1. setting up a simulation that uses an external tool to source spikes.
2. the in-band control and spike exchange protocols. 
3. use of the supplemental tools exposed by Arbor to handle these connections.

Please run it with at least two MPI processes.

Also included is a cross language client (Python)/ server (C++) example for
demonstration. Not yet tested due to OpenMPI fickleness, but a way to use this
could be the following.

In one shell, start the server. It will print some connection information
``` bash
> mpirun -np 2 server
********
```
In another shell start the client, passing the printed information
``` bash
> mpirun -np 2 python client.py ********
```
When using OpenMPI, an additional program needs to be started and its connection 
must be handed to both client and server. Please follow the instructions printed
on screen and/or the documentation.
