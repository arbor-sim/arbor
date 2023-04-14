import unittest
import arbor
from . import fixtures

_mpi_enabled = arbor.__config__


@fixtures.context()
def skipIfNotDistributed(context):
    skipSingleNode = unittest.skipIf(
        context.ranks < 2, "Skipping distributed test on single node."
    )
    skipNotEnabled = unittest.skipIf(
        not _mpi_enabled, "Skipping distributed test, no MPI support in arbor."
    )

    def skipper(f):
        return skipSingleNode(skipNotEnabled(f))

    return skipper


@fixtures.context()
def skipIfDistributed(context):
    return unittest.skipIf(
        context.ranks > 1, "Skipping single node test on multiple nodes."
    )
