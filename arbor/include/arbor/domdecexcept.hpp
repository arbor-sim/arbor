#pragma once

#include <string>

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/domain_decomposition.hpp>

namespace arb {
struct ARB_ARBOR_API dom_dec_exception: public arbor_exception {
    dom_dec_exception(const std::string& what): arbor_exception("Invalid domain decomposition: " + what) {}
};

struct ARB_ARBOR_API invalid_gj_cell_group: dom_dec_exception {
    invalid_gj_cell_group(cell_gid_type gid_0, cell_gid_type gid_1);
    cell_gid_type gid_0, gid_1;
};

struct ARB_ARBOR_API invalid_sum_local_cells: dom_dec_exception {
    invalid_sum_local_cells(unsigned gc_wrong, unsigned gc_right);
    unsigned gc_wrong, gc_right;
};

struct ARB_ARBOR_API duplicate_gid: dom_dec_exception {
    duplicate_gid(cell_gid_type gid);
    cell_gid_type gid;
};

struct ARB_ARBOR_API out_of_bounds: dom_dec_exception {
    out_of_bounds(cell_gid_type gid, unsigned num_cells);
    cell_gid_type gid;
    unsigned num_cells;
};

struct ARB_ARBOR_API invalid_backend: dom_dec_exception {
    invalid_backend(int rank);
    int rank;
};

struct ARB_ARBOR_API incompatible_backend: dom_dec_exception {
    incompatible_backend(int rank, cell_kind kind);
    int rank;
    cell_kind kind;
};
} // namespace arb

