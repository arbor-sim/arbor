import json

def from_json(o, key):
    if key in o:
        return o[key]
    else:
        raise Exception(str('parameter "'+ key+ '" not in input file'))

class cell_parameters:
    def __repr__(self):
        s = "cell parameters\n" \
            "  depth        : {0:10d}\n" \
            "  branch prob  :     [{1:5.2f} : {2:5.2f}]\n" \
            "  compartments :     [{3:5d} : {4:5d}]\n" \
            "  lengths      :     [{5:5.1f} : {6:5.1f}]\n" \
            .format(self.max_depth,
                    self.branch_probs[0], self.branch_probs[1],
                    self.compartments[0], self.compartments[1],
                    self.lengths[0], self.lengths[1])
        return s

    def __init__(self, data=None):
        if data:
            self.max_depth    = from_json(data, 'depth')
            self.branch_probs = from_json(data, 'branch-probs')
            self.compartments = from_json(data, 'compartments')
            self.lengths      = from_json(data, 'lengths')
            self.synapses     = from_json(data, 'synapses')
        else:
            self.max_depth    = 5
            self.branch_probs = [1.0, 0.5]
            self.compartments = [20, 2]
            self.lengths      = [200, 20]
            self.synapses     = 1

class model_parameters:
    def __repr__(self):
        s = "parameters\n" \
            "  name         : {0:>10s}\n" \
            "  cells        : {1:10d}\n" \
            "  duration     : {2:10.0f} ms\n" \
            "  min delay    : {3:10.0f} ms\n" \
            .format(self.name, self.num_cells, self.duration, self.min_delay)
        s+= str(self.cell)
        return s

    def __init__(self, filename=None):
        if filename:
            with open(filename) as f:
                data = json.load(f)
                self.name      = from_json(data, 'name')
                self.num_cells = from_json(data, 'num-cells')
                self.duration  = from_json(data, 'duration')
                self.min_delay = from_json(data, 'min-delay')
                self.cell      = cell_parameters(data)

        else:
            self.name      = 'default'
            self.num_cells = 2
            self.duration  = 100
            self.min_delay = 10
            self.cell = cell_parameters()
