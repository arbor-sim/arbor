#pragma once

#include <any>

#include <arbor/arbexcept.hpp>
#include <arbor/util/hopefully.hpp>

namespace arb {

struct label_parse_error: arb::arbor_exception {
    label_parse_error(const std::string& msg);
};

template <typename T>
using parse_hopefully = arb::util::hopefully<T, label_parse_error>;

parse_hopefully<std::any> parse_label_expression(const std::string&);
bool valid_label_name(const std::string &in);


} // namespace arb
