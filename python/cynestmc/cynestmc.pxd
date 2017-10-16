from libcpp cimport bool as cbool
from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

cdef extern from "common_types.hpp" namespace "nest::mc":
    ctypedef uint32_t cell_gid_type
    ctypedef uint32_t cell_lid_type
    ctypedef cell_gid_type cell_size_type
    ctypedef float time_type
    
    ctypedef enum cell_kind
        cable1d_neuron
        
    cdef struct cell_member_type:
        cell_gid_type gid
        cell_lid_type index

cdef extern from "backends.hpp" namespace "nest::mc":
    cdef enum backend_policy:
        use_multicore,
        prefer_gpu

cdef extern from "domain_decomposition.hpp" namespace "nest::mc":
    cdef struct group_rules:
        cell_size_type target_group_size
        backend_policy policy

    cdef cppclass domain_decomposition:
        domain_decomposition(const CRecipe&, const group_rules&) except+

cdef extern from "pyrecipe.hpp" namespace "nest::mc":
    ctypedef cell_member_type cell_connection_endpoint
    
    cdef struct cell_connection:
        cell_connection_endpoint source
        cell_connection_endpoint dest
        float weight
        float delay
        
    cdef cppclass CRecipe "nest::mc::recipe":
        cell_size_type num_cells() except+
        util::unique_any get_cell(cell_gid_type)
        cell_kind get_cell(cell_gid_type) except+
        cell_count_info get_cell_count_info(cell_gid_type) except+
        vector[cell_connection] connections_on(cell_gid_type) except+

cdef extern from "model.hpp" namespace "nest::mc":
    cdef cppclass CModel "nest::mc::model":
        CModel(const CRecipe&, ...) except+
        void reset() except+
        time_type run(time_type tfinal, time_type dt) except+
        size_t num_spikes() except+
