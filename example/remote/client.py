#!/usr/bin/env python3

from builtins import exit
import arbor as A
from mpi4py import MPI
import sys

class recipe(A.recipe):
    def __init__(self, num=1):
        A.recipe.__init__(self)
        self.num = num
        self.weight = 20.0
        self.delay = 0.1

    def num_cells(self):
        return self.num

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def cell_description(self, _):
        cell = A.lif_cell("src", "tgt")
        cell.tau_m =   2.0;
        cell.V_th  = -10.0;
        cell.C_m   =  20.0;
        cell.E_L   = -23.0;
        cell.V_m   = -23.0;
        cell.E_R   = -23.0;
        cell.t_ref =   0.2;
        return cell

    def external_connections_on(self, _):
        return [A.external_connection(rid,
                                      "tgt",
                                      self.weight,
                                      self.delay)
                for rid in range(10)]

    def get_probes(self, _):
        return [A.lif_probe_voltage()]

_, secret = sys.argv

mpi = MPI.COMM_WORLD
inter = mpi.Connect(secret)

ctx = A.context(mpi=mpi, inter=inter)
rec = recipe()
sim = A.simulation(rec, ctx)
mid = sim.min_delay()
print(f'[ARB] min delay={mid}')
sim.run(T, dt)
