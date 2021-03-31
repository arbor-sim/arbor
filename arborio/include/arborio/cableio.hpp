#pragma once

#include <arbor/cable_cell.hpp>
#include <arbor/s_expr.hpp>

namespace arborio {
std::string acc_version();

struct cableio_parse_error: arb::arbor_exception {
    explicit cableio_parse_error(const std::string& msg, const arb::src_location& loc);
};

struct cableio_morphology_error: arb::arbor_exception {
    explicit cableio_morphology_error(const unsigned bid);
};

struct cableio_version_error: arb::arbor_exception {
    explicit cableio_version_error(const std::string& version);
};

template <typename T>
using parse_hopefully = arb::util::expected<T, cableio_parse_error>;
using cable_cell_variant = std::variant<arb::morphology, arb::label_dict, arb::decor, arb::cable_cell>;

struct meta_data {
    std::string version = acc_version();
};
struct cable_cell_component {
    meta_data meta;
    cable_cell_variant component;
};

std::ostream& write_component(std::ostream&, const cable_cell_component&);
std::ostream& write_component(std::ostream&, const arb::decor& x, const meta_data& m = {});
std::ostream& write_component(std::ostream&, const arb::label_dict& x, const meta_data& m = {});
std::ostream& write_component(std::ostream&, const arb::morphology& x, const meta_data& m = {});
std::ostream& write_component(std::ostream&, const arb::cable_cell& x, const meta_data& m = {});

parse_hopefully<cable_cell_component> parse_component(const std::string&);
parse_hopefully<cable_cell_component> parse_component(std::istream&);

} // namespace arborio