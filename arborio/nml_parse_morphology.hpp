#pragma once

#include <arborio/neuroml.hpp>
#include <arborio/loaded_morphology.hpp>

#include <pugixml.hpp>

namespace arborio {

loaded_morphology nml_parse_morphology_element(const pugi::xml_node& morph, enum neuroml_options::values);

} // namespace arborio
