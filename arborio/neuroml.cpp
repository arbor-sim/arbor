#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arborio/neuroml.hpp>

#include "nml_parse_morphology.hpp"
#include "xmlwrap.hpp"

using std::optional;
using std::nullopt;

using namespace arborio::xmlwrap;

namespace arborio {

static std::string fmt_error(const char* prefix, const std::string& err, unsigned line) {
    return prefix + (line==0? err: "line " + std::to_string(line) + ": " + err);
}

nml_no_document::nml_no_document():
    neuroml_exception("no NeuroML document to parse")
{}

nml_parse_error::nml_parse_error(const std::string& error_msg, unsigned line):
    neuroml_exception(fmt_error("parse error: ", error_msg, line)),
    error_msg(error_msg),
    line(line)
{}

nml_bad_segment::nml_bad_segment(unsigned long long segment_id, unsigned line):
    neuroml_exception(
        fmt_error(
            "bad morphology segment: ",
            "segment "+(segment_id+1==0? "unknown": "\""+std::to_string(segment_id)+"\""),
            line)),
    segment_id(segment_id),
    line(line)
{}

nml_bad_segment_group::nml_bad_segment_group(const std::string& group_id, unsigned line):
    neuroml_exception(
        fmt_error(
            "bad morphology segmentGroup: ",
            "segmentGroup id "+(group_id.empty()? "unknown": "\""+group_id+"\""),
            line)),
    group_id(group_id),
    line(line)
{}

nml_cyclic_dependency::nml_cyclic_dependency(const std::string& id, unsigned line):
    neuroml_exception(
        fmt_error(
            "cyclic dependency: ",
            "element id \""+id+"\"",
            line)),
    id(id),
    line(line)
{}

struct neuroml_impl {
    xml_doc doc;

    neuroml_impl() {}

    explicit neuroml_impl(std::string text) {
        xml_error_scope err;
        doc = xml_doc(text);
    }

    xml_xpathctx make_context() const {
        if (!doc) throw nml_no_document{};

        auto ctx = xpath_context(doc);
        ctx.register_ns("nml", "http://www.neuroml.org/schema/neuroml2");
        return ctx;
    }
};

neuroml::neuroml(): impl_(new neuroml_impl) {}
neuroml::neuroml(std::string nml_document): impl_(new neuroml_impl{nml_document}) {}

neuroml::neuroml(neuroml&&) = default;
neuroml& neuroml::operator=(neuroml&&) = default;

neuroml::~neuroml() = default;

std::vector<std::string> neuroml::cell_ids() const {
    xml_error_scope err;
    std::vector<std::string> result;

    auto ctx = impl_->make_context();
    auto matches = ctx.query("//nml:neuroml/nml:cell/@id");

    result.reserve(matches.size());
    for (auto node: matches) {
        result.push_back(std::string(node.content()));
    }

    return result;
}

std::vector<std::string> neuroml::morphology_ids() const {
    xml_error_scope err;
    std::vector<std::string> result;

    auto ctx = impl_->make_context();
    auto matches = ctx.query("//nml:neuroml/nml:morphology/@id");

    result.reserve(matches.size());
    for (auto node: matches) {
        result.push_back(std::string(node.content()));
    }

    return result;
}

optional<nml_morphology_data> neuroml::morphology(const std::string& morph_id, enum neuroml_options::values options) const {
    xml_error_scope err;
    auto ctx = impl_->make_context();
    auto matches = ctx.query("//nml:neuroml/nml:morphology[@id="+xpath_escape(morph_id)+"]");

    return matches.empty()? nullopt: optional(nml_parse_morphology_element(ctx, matches[0], options));
}

optional<nml_morphology_data> neuroml::cell_morphology(const std::string& cell_id, enum neuroml_options::values options) const {
    xml_error_scope err;
    auto ctx = impl_->make_context();
    auto matches = ctx.query(
        "( //nml:neuroml/nml:morphology[@id=string((//nml:neuroml/nml:cell[@id="+xpath_escape(cell_id)+"]/@morphology)[1])] | "
        "  //nml:neuroml/nml:cell[@id="+xpath_escape(cell_id)+"]/nml:morphology )[1]");

    if (matches.empty()) return nullopt;

    nml_morphology_data M = nml_parse_morphology_element(ctx, matches[0], options);
    M.cell_id = cell_id;
    return M;
}

} // namespace arborio
