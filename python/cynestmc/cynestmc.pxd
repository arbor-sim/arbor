from libcpp cimport bool as cbool
from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

cdef extern from "common_types.hpp" namespace "nest::mc":
    ctypedef uint32_t cell_gid_type
    ctypedef uint32_t cell_lid_type
    ctypedef cell_gid_type cell_size_type
    
    ctypedef enum cell_kind
        cable1d_neuron
        
    cdef struct cell_member_type:
        cell_gid_type gid
        cell_lid_type index

cdef extern from "recipe.hpp" namespace "nest::mc":
    ctypedef cell_member_type cell_connection_endpoint
    
    cdef struct cell_connection:
        cell_connection_endpoint source
        cell_connection_endpoint dest
        float weight
        float delay
        
    cdef cppclass Crecipe "nest::mc::recipe":
        cell_size_type num_cells() except+
        # util::unique_any get_cell(cell_gid_type)
        cell_kind get_cell(cell_gid_type) except+
        cell_count_info get_cell_count_info(cell_gid_type) except+
        vector[cell_connection] connections_on(cell_gid_type) except+

cdef int c_call_something(cyrecipe*) except+
