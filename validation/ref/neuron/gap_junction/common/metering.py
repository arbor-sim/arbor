from timeit import default_timer as timer
import socket
import json

class meter:
    def __repr__(self):
        s = "-- meters ------------------------------------\n" \
            "{0:20s}{1:>20s}\n" \
            "----------------------------------------------\n" \
            .format("region", "time (s)")

        for i in range(len(self.checkpoints)):
            s += "{0:20s}{1:20.5f}\n".format(self.checkpoints[i], self.times[i])

        return s

    def __init__(self, with_mpi=False):
        self.checkpoints = []
        self.times = []
        self.running = False
        self.with_mpi = with_mpi

        if self.with_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        self.timepoint = timer()

    def start(self):
        if self.with_mpi:
            self.comm.Barrier()
        self.timepoint = timer()

    def checkpoint(self, name):
        if self.with_mpi:
            self.comm.Barrier()
        end = timer()
        self.times.append(end-self.timepoint)
        self.checkpoints.append(name)
        self.timepoint = timer()

    def print(self):
        if self.with_mpi:
            if self.comm.rank==0:
                print(self)

class meter_report:
    def __init__(self, checkpoints, times, num_domains):
        self.checkpoints = checkpoints
        self.times = times
        self.num_domains = num_domains


    def to_json(self):
        output = { 'checkpoints': self.checkpoints,
                   'num_domains': self.num_domains,
                   'meters': [{'name': 'time',    'units': 's',   'measurements': self.times}]}

        return json.dumps(output)

    def to_file(self, filename):
        fid = open(filename, 'w')
        fid.write(self.to_json())
        fid.write('\n')
        fid.close()

def report_from_meter(m):
    times = []
    ndom = 1
    if m.with_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        ndom = size
        # gather all the times
        for t in m.times:
            all_times = comm.gather(t, root=0)
            if rank==0:
                times.append(all_times)

    else:
        for t in m.times:
            times.append([t])

    return meter_report(m.checkpoints, times, ndom)

def report_from_json(j):
    js = json.loads(j)
    for record in js['meters']:
        if record['name']=='time':
            times = record['measurements']

    return meter_report(js['checkpoints'], times, js['num_domains'])

