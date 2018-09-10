import pyarb as arb

#
#   method 1: use helpers provided by our Python wrapper
#

#arb.mpi_init();

#comm = arb.mpi_comm()
#print(comm)

#
#   method 2: get MPI_Comm from mpi4py
#
import mpi4py.MPI as mpi
comm = arb.mpi_comm_from_mpi4py(mpi.COMM_WORLD)
print(comm)

resources = arb.proc_allocation()

context = arb.context(resources, comm)

print(context)

arb.mpi_finalize();
