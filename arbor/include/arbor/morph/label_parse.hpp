#pragma once

#include <arbor/arbexcept.hpp>
#include <arbor/util/hopefully.hpp>
#include <arbor/util/any.hpp>

namespace pyarb {

struct label_parse_error: arb::arbor_exception {
    label_parse_error(const std::string& msg);
};

template <typename T>
using parse_hopefully = arb::util::hopefully<T, label_parse_error>;

parse_hopefully<arb::util::any> parse_label_expression(const std::string&);
bool test_identifier(const std::string &in);


} // namespace pyarb
