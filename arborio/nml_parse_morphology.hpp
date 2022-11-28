#pragma once

#include <arborio/neuroml.hpp>

#include <pugixml.hpp>

namespace arborio {

nml_morphology_data nml_parse_morphology_element(const pugi::xml_node& morph, enum neuroml_options::values);

} // namespace arborio
