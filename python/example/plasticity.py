import arbor as A


class recipe(A.recipe):
    def __init__(self, n):
        # Call our parent; needed for proper initialization.
        A.recipe.__init__(self)
        # Cell count; should be at least 3 for the example to work.
        assert n >= 3
        self.cells = n
        # Uniform weights and delays for the connectivity.
        self.weight = 0.75
        self.delay = 0.1
        # Track the connections from the source cell to a given gid.
        self.connected = set()

    def num_cells(self):
        return self.cells

    def cell_kind(self, gid):
        # Cell 0 is the spike source, all others cable cells.
        if gid == 0:
            return A.cell_kind.spike_source
        else:
            return A.cell_kind.cable

    def global_properties(self, kind):
        # Cable cells return the NRN defaults.
        if kind == A.cell_kind.cable:
            return A.neuron_cable_properties()
        # Spike source cells have nothing to report.
        return None

    def cell_description(self, gid):
        # Cell 0 is reserved for the spike source, spiking every 0.0125ms.
        if gid == 0:
            return A.spike_source_cell("source", A.regular_schedule(0.0125))
        # All cells >= 1 are cable cells w/ a simple, soma-only morphology
        # comprising two segments of radius r=3um and length l=3um *each*.
        #
        #   +-- --+  ^
        #   |  *  |  3um
        #   +-- --+  v
        #   < 6um >
        #
        #  * Detectors and Synapses are here, at the midpoint.
        r = 3
        tree = A.segment_tree()
        tree.append(A.mnpos, A.mpoint(-r, 0, 0, r), A.mpoint(r, 0, 0, r), tag=1)
        # ... and a synapse/detector pair
        decor = A.decor()
        #   - just have a leaky membrane here.
        decor.paint("(all)", A.density("pas"))
        #   - synapse to receive incoming spikes from the source cell.
        decor.place("(location 0 0.5)", A.synapse("expsyn"), "synapse")
        #   - detector for reporting spikes on the cable cells.
        decor.place("(location 0 0.5)", A.threshold_detector(-10.0), "detector")
        # return the cable cell description
        return A.cable_cell(tree, decor)

    def connections_on(self, gid):
        # If we have added a connection to this cell, return it, else nothing.
        if gid in self.connected:
            # Connect spike source to synapse(s)
            return [A.connection((0, "source"), "synapse", self.weight, self.delay)]
        return []

    def add_connection_to_spike_source(self, to):
        """Add a connection from the spike source at gid=0 to the cable cell
        gid=<to>. Note that we try to minimize the information stored here,
        which is important with large networks. Also we cannot connect the
        source back to itself.
        """
        assert to != 0
        self.connected.add(to)


# Context for multi-threading
ctx = A.context(threads=2)
# Make an unconnected network with 2 cable cells and one spike source,
rec = recipe(3)
# but before setting up anything, connect cable cell gid=1 to spike source gid=0
# and make the simulation of the simple network
#
#    spike_source <gid=0> ----> cable_cell <gid=1>
#
#                               cable_cell <gid=2>
#
# Note that the connection is just _recorded_ in the recipe, the actual connectivity
# is set up in the simulation construction.
rec.add_connection_to_spike_source(1)
sim = A.simulation(rec, ctx)
sim.record(A.spike_recording.all)
# then run the simulation for a bit
sim.run(0.25, 0.025)
# update the simulation to
#
#    spike_source <gid=0> ----> cable_cell <gid=1>
#                        \
#                         ----> cable_cell <gid=2>
rec.add_connection_to_spike_source(2)
sim.update(rec)
# and run the simulation for another bit.
sim.run(0.5, 0.025)
# when finished, print spike times and locations.
source_spikes = 0
print("Spikes:")
for (gid, lid), t in sim.spikes():
    if gid == 0:
        source_spikes += 1
    else:
        print(f"  * {t:>8.4f}ms: gid={gid} detector={lid}")
print(f"Source spiked {source_spikes:>5d} times.")
