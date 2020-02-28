#pragma once

#include <string>

#include <arbor/arbexcept.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

struct morphology_error: public arbor_exception {
    morphology_error(const std::string& what): arbor_exception(what) {}
};

struct invalid_mlocation: morphology_error {
    invalid_mlocation(mlocation loc);
    mlocation loc;
};

struct no_such_branch: morphology_error {
    no_such_branch(msize_t bid);
    msize_t bid;
};

struct invalid_mcable: morphology_error {
    invalid_mcable(mcable cable);
    mcable cable;
};

struct invalid_mcable_list: morphology_error {
    invalid_mcable_list();
};

struct invalid_sample_parent: morphology_error {
    invalid_sample_parent(msize_t parent, msize_t tree_size);
    msize_t parent;
    msize_t tree_size;
};

struct label_type_mismatch: morphology_error {
    label_type_mismatch(const std::string& label);
    std::string label;
};

struct incomplete_branch: morphology_error {
    incomplete_branch(msize_t bid);
    msize_t bid;
};

struct unbound_name: morphology_error {
    unbound_name(const std::string& name);
    std::string name;
};

struct circular_definition: morphology_error {
    circular_definition(const std::string& name);
    std::string name;
};

} // namespace arb

