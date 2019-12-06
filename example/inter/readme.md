# Arbor: Inter

Arbor miniapp that allows receiving of spikes from
and external simulator (NEST).
This is a first Proof Of Principle implementation with a tight
coupling between Arbor and NEST. 

arb_inter: real arbor app 
arb-proxy: pretend arbor app
arb-proxy.py: pretend arbor app, pure python for inclusion in NEST3 
nest_proxy: pretend nest app
nest_sender.py: real nest app (needs installed NEST3, with arbor back-end)

Todo:
- MPI world is still shared, this might be changed to a MPI client server structure
- Spikes exchanged are NEURON format: <Gid, lid, Float>. This is not needed for NEST
- Code duplication of mpi_util in NEST and Neuron. This risks these going out of sync.
- MPI hand shake is hard-coded. No version test is done.

w.klijn@fz-juelich.de for co-simulation related questions.