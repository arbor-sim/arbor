#pragma once

#include <arborio/cableio.hpp>

namespace arborio {

parse_hopefully<std::any> parse_expression(const std::string&);

} // namespace arborio
