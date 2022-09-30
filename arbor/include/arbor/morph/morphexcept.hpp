#pragma once

#include <string>

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

struct ARB_SYMBOL_VISIBLE morphology_error: public arbor_exception {
    morphology_error(const std::string& what): arbor_exception(what) {}
};

struct ARB_SYMBOL_VISIBLE invalid_mlocation: morphology_error {
    invalid_mlocation(mlocation loc);
    mlocation loc;
};

struct ARB_SYMBOL_VISIBLE no_such_branch: morphology_error {
    no_such_branch(msize_t bid);
    msize_t bid;
};

struct ARB_SYMBOL_VISIBLE no_such_segment: arbor_exception {
    explicit no_such_segment(msize_t sid);
    msize_t sid;
};

struct ARB_SYMBOL_VISIBLE invalid_mcable: morphology_error {
    invalid_mcable(mcable cable);
    mcable cable;
};

struct ARB_SYMBOL_VISIBLE invalid_mcable_list: morphology_error {
    invalid_mcable_list();
};

struct ARB_SYMBOL_VISIBLE invalid_segment_parent: morphology_error {
    invalid_segment_parent(msize_t parent, msize_t tree_size);
    msize_t parent;
    msize_t tree_size;
};

struct ARB_SYMBOL_VISIBLE duplicate_stitch_id: morphology_error {
    duplicate_stitch_id(const std::string& id);
    std::string id;
};

struct ARB_SYMBOL_VISIBLE no_such_stitch: morphology_error {
    no_such_stitch(const std::string& id);
    std::string id;
};

struct ARB_SYMBOL_VISIBLE missing_stitch_start: morphology_error {
    missing_stitch_start(const std::string& id);
    std::string id;
};

struct ARB_SYMBOL_VISIBLE invalid_stitch_position: morphology_error {
    invalid_stitch_position(const std::string& id, double along);
    std::string id;
    double along;
};

struct ARB_SYMBOL_VISIBLE label_type_mismatch: morphology_error {
    label_type_mismatch(const std::string& label);
    std::string label;
};

struct ARB_SYMBOL_VISIBLE incomplete_branch: morphology_error {
    incomplete_branch(msize_t bid);
    msize_t bid;
};

struct ARB_SYMBOL_VISIBLE unbound_name: morphology_error {
    unbound_name(const std::string& name);
    std::string name;
};

struct ARB_SYMBOL_VISIBLE circular_definition: morphology_error {
    circular_definition(const std::string& name);
    std::string name;
};

} // namespace arb

