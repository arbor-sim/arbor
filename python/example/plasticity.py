import arbor as A

class recipe(A.recipe):
    def __init__(self, n):
        super().__init__()
        self.ccp = A.neuron_cable_properties()
        self.connected = {}
        self.n =n

    def num_cells(self):
        return self.n

    def cell_kind(self, gid):
        if gid == 0:
            return A.cell_kind.spike_source
        else:
            return A.cell_kind.cable

    def global_properties(self, kind):
        if kind == A.cell_kind.cable:
            return self.ccp
        return None

    def cell_description(self, gid):
        if gid == 0:
            return A.spike_source_cell("source", A.regular_schedule(0.0125))

        r = 3
        tree = A.segment_tree()
        tree.append(A.mnpos, A.mpoint(-r, 0, 0, r), A.mpoint(r, 0, 0, r), tag=1)
        decor = A.decor()
        decor.paint("(all)", A.density("pas"))
        decor.place("(location 0 0.5)", A.synapse("expsyn"), "synapse")
        decor.place("(location 0 0.5)", A.spike_detector(-10.0), "detector")
        return A.cable_cell(tree, A.label_dict(), decor)

    def connections_on(self, gid):
        if gid in self.connected:
            w, d = self.connected[gid]
            return [A.connection((0, "source"), "synapse", w, d)]
        return []

    def add_connection(self, to):
        self.connected[to] = (0.75, 0.1)


rec = recipe(3)
rec.add_connection(1)
sim = A.simulation(rec)
sim.record(A.spike_recording.all)

sim.run(0.25, 0.025)

rec.add_connection(2)
sim.update_connections(rec)

sim.run(0.5, 0.025)

print("spikes:")
for sp in sim.spikes():
    print(" ", sp)
