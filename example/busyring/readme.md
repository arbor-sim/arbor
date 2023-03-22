# Busyring Benchmark

A simple benchmark of spiking cells in multiple rings. Each ring propagates a
single spike indefinitely. Rings are interconnected using zero-weight synapses
as to stress the MPI parts while not altering the ring steady-state. Cells come
in two varieties: simple and complex. Complex cells are based on the Allen
example and are extremely heavy on compute.

Example configs show an initialization only benchmark with 2048 complex cells
and one with a larger runtime with 128 complex cells.
