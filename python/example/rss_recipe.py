import pyarb as arb

class rss_recipe(arb.recipe):
    def num_cells(self):
        return 4

    def cell_description(self, gid):
        cell = arb.rss_cell()
        cell.start_time = 0
        cell.period = 2
        cell.stop_time = 10
        return cell

    def kind(self, gid):
        return arb.cell_kind.regular_spike

node = arb.get_node_info()
recipe = rss_recipe()

decomp = arb.partition_load_balance(recipe, node)

model = arb.model(recipe, decomp)
recorder = arb.make_spike_recorder(model)
model.run(100, 0.01)

print(recorder.spikes)
