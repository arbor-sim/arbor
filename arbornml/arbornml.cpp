#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arbornml/arbornml.hpp>
#include <arbornml/nmlexcept.hpp>

#include "parse_morphology.hpp"
#include "xmlwrap.hpp"

using std::optional;
using std::nullopt;

namespace arbnml {

struct neuroml_impl {
    xml_doc doc;

    neuroml_impl() {}

    explicit neuroml_impl(std::string text) {
        xml_error_scope err;
        doc = xml_doc(text);
    }

    xml_xpathctx make_context() const {
        if (!doc) throw no_document{};

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

optional<morphology_data> neuroml::morphology(const std::string& morph_id) const {
    xml_error_scope err;
    auto ctx = impl_->make_context();
    auto matches = ctx.query("//nml:neuroml/nml:morphology[@id="+xpath_escape(morph_id)+"]");

    return matches.empty()? nullopt: optional(parse_morphology_element(ctx, matches[0]));
}

optional<morphology_data> neuroml::cell_morphology(const std::string& cell_id) const {
    xml_error_scope err;
    auto ctx = impl_->make_context();
    auto matches = ctx.query(
        "( //nml:neuroml/nml:morphology[@id=string((//nml:neuroml/nml:cell[@id="+xpath_escape(cell_id)+"]/@morphology)[1])] | "
        "  //nml:neuroml/nml:cell[@id="+xpath_escape(cell_id)+"]/nml:morphology )[1]");

    if (matches.empty()) return nullopt;

    morphology_data M = parse_morphology_element(ctx, matches[0]);
    M.cell_id = cell_id;
    return M;
}

} // namespace arbnml
