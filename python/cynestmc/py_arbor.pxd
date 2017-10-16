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

cdef extern from "model.hpp":
    model(const recipe&, const domain_decomposition&) except+
    void reset() except+
    time_type run(time_type tfinal, time_type dt) except+
    std::size_t num_spikes() except+
    std::size_t num_groups() except+
    std::size_t num_cells()  except+
