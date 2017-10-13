from libc.stdlib cimport malloc, free

def py_arbor(argv):
    cdef int argc = <int> len(argv)
    cdef char** argv = <char**> malloc((argc+1)*sizeof(char*))

    cdef io::cloptions options=io::read_options(argc, argv, global_policy::id()==0)
    cdef probe_distribution pdist
    cdef group_rules rules
    recipe = make_recipe（options，pdist）

    decomp = domain_decomposition(*recipe, rules)
    modle m(*recipe, decomp)
    m.run(options.tfinal, options.dt)
    std::cout <<m.num_spikes()



