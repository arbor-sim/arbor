std::unique_ptr<recipe> make_recipe(int num_cells) {
    // Put pdist construction here
    // const probe_distribution& pdist
    // later, other options const io::cl_options& options outside

    basic_recipe_param p;

    if (options.morphologies) {
        std::cout << "loading morphologies...\n";
        p.morphologies.clear();
        load_swc_morphology_glob(p.morphologies, options.morphologies.get());
        std::cout << "loading morphologies: " << p.morphologies.size() << " loaded.\n";
    }
    p.morphology_round_robin = options.morph_rr;

    p.num_compartments = options.compartments_per_segment;
    p.num_synapses = options.all_to_all? options.cells-1: options.synapses_per_cell;
    p.synapse_type = options.syn_type;

    if (options.all_to_all) {
        return make_basic_kgraph_recipe(options.cells, p, pdist);
    }
    else if (options.ring) {
        return make_basic_ring_recipe(options.cells, p, pdist);
    }
    else {
        return make_basic_rgraph_recipe(options.cells, p, pdist);
    }
}
