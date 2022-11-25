#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arborio/neuroml.hpp>

#include "nml_parse_morphology.hpp"
#include <pugixml.hpp>

#include <iostream>

using std::optional;
using std::nullopt;

namespace arborio {

static std::string fmt_error(const char* prefix, const std::string& err) {
    return prefix + err;
}

nml_no_document::nml_no_document():
    neuroml_exception("no NeuroML document to parse")
{}

nml_parse_error::nml_parse_error(const std::string& error_msg):
    neuroml_exception(fmt_error("parse error: ", error_msg)),
    error_msg(error_msg)
{}

nml_bad_segment::nml_bad_segment(unsigned long long segment_id):
    neuroml_exception(
        fmt_error(
            "bad morphology segment: ",
            "segment "+(segment_id+1==0? "unknown": "\""+std::to_string(segment_id)+"\""))),
    segment_id(segment_id)
{}

nml_bad_segment_group::nml_bad_segment_group(const std::string& group_id):
    neuroml_exception(
        fmt_error(
            "bad morphology segmentGroup: ",
            "segmentGroup id "+(group_id.empty()? "unknown": "\""+group_id+"\""))),
    group_id(group_id)
{}

nml_cyclic_dependency::nml_cyclic_dependency(const std::string& id):
    neuroml_exception(
        fmt_error(
            "cyclic dependency: ",
            "element id \""+id+"\"")),
    id(id)
{}

struct ARB_ARBORIO_API neuroml_impl {
    pugi::xml_document doc;
    std::string raw;

    neuroml_impl() {}

    explicit neuroml_impl(std::string text): raw{text} {
        auto res = doc.load_buffer_inplace(raw.data(), raw.size()+1);
        if (res.status != pugi::xml_parse_status::status_ok) {
            throw nml_parse_error{res.description()};
        }
    }
};

neuroml::neuroml(): impl_(new neuroml_impl) {}
neuroml::neuroml(std::string nml_document): impl_(new neuroml_impl{nml_document}) {}

neuroml::neuroml(neuroml&&) = default;
neuroml& neuroml::operator=(neuroml&&) = default;

neuroml::~neuroml() = default;

std::vector<std::string> neuroml::cell_ids() const {
    auto matches = impl_->doc.select_nodes("//neuroml/cell/@id");
    std::vector<std::string> result;
    result.reserve(matches.size());
    for (const auto& it: matches) {
        result.push_back(it.attribute().as_string());
    }
    return result;
}

std::vector<std::string> neuroml::morphology_ids() const {
    auto matches = impl_->doc.select_nodes("//neuroml/morphology/@id");
    std::vector<std::string> result;
    result.reserve(matches.size());
    for (const auto& it: matches) {
        result.push_back(it.attribute().as_string());
    }
    return result;
}

inline std::string xpath_escape(const std::string& x) {
    auto npos = std::string::npos;
     if (x.find_first_of("'")==npos) {
         return "'"+x+"'";
     }
     else if (x.find_first_of("\"")==npos) {
         return "\""+x+"\"";
     }
     else {
         std::string r = "concat(";
         std::string::size_type i = 0;
         for (;;) {
             auto j = x.find_first_of("'", i);
             r += "'";
             r.append(x, i, j==npos? j: j-i);
             r += "'";
             if (j==npos) break;
             r += ",\"";
             i = j+1;
             j = x.find_first_not_of("'",i);
             r.append(x, i, j==npos? j: j-i);
             r += "\"";
             if (j==npos) break;
             r += ",";
             i = j+1;
         }
         r += ")";
         return r;
     }
}

optional<nml_morphology_data> neuroml::morphology(const std::string& morph_id, enum neuroml_options::values options) const {
    auto id = xpath_escape(morph_id);
    auto query = "//neuroml/morphology[@id=" + id + "]";
    auto match = impl_->doc.select_node(query.data()).node();
    if (match.empty()) return {};
    return nml_parse_morphology_element(match, options);
}

optional<nml_morphology_data> neuroml::cell_morphology(const std::string& cell_id, enum neuroml_options::values options) const {
    auto id = xpath_escape(cell_id);
    auto query = "( //neuroml/morphology[@id=string((//neuroml/cell[@id=" + id + "]/@morphology)[1])] | "
        "  //neuroml/cell[@id=" + id + "]/morphology )[1]";
    auto match = impl_->doc.select_node(query.data()).node();
    if (match.empty()) return nullopt;
    nml_morphology_data M = nml_parse_morphology_element(match, options);
    M.cell_id = cell_id;
    return M;
}

} // namespace arborio
