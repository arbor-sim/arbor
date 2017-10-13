# This file is used to declear the parameters and functions that will be used.

#import the types from c++ library
from libcpp.vector cimport vector

cdef extern from "common_types.hpp" namespace "nest::mc":
    ctypedef std::uint32_t cell_gid_type    #TODO
    ctypedef cell_gid_type cell_size_type
    ctypedef float time_type
    ctypedef std::uint32_t cell_lid_type

    ctypedef enum cell_kind:
        cable1d_neuron

    ctypedef struct cell_member_type:
        cell_gid_type gid
        cell_lid_type index

cdef extern from "backend_policy" namespace "nest::mc":
   # cdef enum class backend_policy:  #TODO how to wrap c++ enum class
    cdef enum  backend_policy:
        use_multicore
        prefer_gpu

cdef extern from "domain_decomposition.hpp" namespace "nest::mc":
    cdef struct group_rules:
        cell_size_type target_group_size
        backend_policy policy

    cdef cppclass domain_decomposition:
        domain_decomposition(const recipe&, const group_rules&) #TODO Is this the right way to re_define?


cdef extern from "recipe.hpp" namespace "nest::mc":
    ctypedef cell_member_type cell_connection_endpoint
    ctypedef struct cell_count_info:
        cell_size_type num_sources
        cell_size_type num_targets
        cell_size_type num_probes

    ctypedef struct cell_connection:
        cell_connection_endpoint source
        cell_connection_endpoint dest
        float weight
        float delay

    cdef cppclass recipe:

         cell_size_type num_cells()
         util::unique_any get_cell(cell_gid_type)    #TODO how to import it?
         cell_kind get_cell_kind(cell_gid_type)
         cell_count_info get_cell_count_info(cell_gid_type)
         std::vector<cell_connection> connections_on(cell_gid_type) #TODO how to fix this

cdef extern from "miniapp_recipes.hpp" namespace "nest::mc":
    cdef struct probe_distribution:
        float proportion = 1.0
        bool all_segments
        bool membrane_voltage
        bool membrane_current

cdef extern from "io.hpp" namespace "nest::mc::io":
    cdef struct cl_options:
        # uint32_t cells = 1000
        # uint32_t synapses_per_cell = 500
        # std::string syn_type = "expsyn"
        # uint32_t compartments_per_segment = 100
        # util::optional<std::string> morphologies
        # bool morph_rr = false
        # bool all_to_all = false
        # bool ring = false
        # double tfinal = 100.
        # double dt = 0.025
        # uint32_t group_size = 1
        # bool bin_regular = false
        # double bin_dt = 0.0025
        # bool probe_soma_only
        # double probe_ratio = 0
        # std::string trace_prefix
        # util::optional<unsigned> trace_max_gid
        # bool spike_file_output = false
        # bool single_file_per_rank = false
        # bool over_write = true
        # std::string output_path = "./"
        # std::string file_name = "spikes"
        # std::string file_extension = "gdf"
        # int dry_run_ranks = 1
        # bool profile_only_zero = false
        # bool report_compartments = false
        # bool verbose = false
    cl_options read_options(int argc, char** argv, bool allow_write = true)


cdef extern from "model.hpp":
    model(const recipe&, const domain_decomposition&) except+
    void reset() except+
    time_type run(time_type tfinal, time_type dt) except+
    std::size_t num_spikes() except+
    std::size_t num_groups() except+
    std::size_t num_cells()  except+
