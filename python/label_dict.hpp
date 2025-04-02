#pragma once

#include <string>
#include <unordered_map>

#include <arborio/label_parse.hpp>
#include <arbor/morph/label_dict.hpp>

#include <pybind11/pybind11.h>

#include "strprintf.hpp"

namespace pyarb {

struct label_dict {
    using str_map = std::unordered_map<std::string, std::string>;
    ::arb::label_dict dict;
    str_map locsets;
    str_map regions;
    str_map iexpressions;

    label_dict() = default;

    label_dict(const str_map& in) {
        for (auto& [k, v]: in) setitem(k, v);
    }

    label_dict& add_swc_tags() {
        setitem("soma", "(tag 1)");
        setitem("axon", "(tag 2)");
        setitem("dend", "(tag 3)");
        setitem("apic", "(tag 4)");
        return *this;
    }
    
    label_dict(const ::arb::label_dict& ld): dict(ld) {
        for (const auto& [k, v]: ld.locsets()) {
            std::stringstream os;
            os << v;
            locsets[k] = os.str();
        }
        for (const auto& [k, v]: ld.regions()) {
            std::stringstream os;
            os << v;
            regions[k] = os.str();
        }
        for (const auto& [k, v]: ld.iexpressions()) {
            std::stringstream os;
            os << v;
            iexpressions[k] = os.str();
        }
    }

    label_dict(label_dict&&) = default;
    label_dict(const label_dict&) = default;

    std::size_t size() const  { return dict.size(); }

    auto& extend(const label_dict& other, std::string prefix = "") {
        dict.extend(other.dict, prefix);
        for (const auto& [k, v]: other.locsets) locsets[prefix + k] = v;
        for (const auto& [k, v]: other.regions) regions[prefix + k] = v;
        for (const auto& [k, v]: other.iexpressions) iexpressions[prefix + k] = v;
        return *this;
    }

    void setitem(const std::string& name, const std::string& desc) {
        using namespace std::string_literals;
        // The following code takes an input name and a region or locset or iexpr
        // description, e.g.:
        //      name='reg', desc='(tag 4)'
        //      name='loc', desc='(terminal)'
        //      name='foo', desc='(join (tag 2) (tag 3))'
        // Then it parses the description, and tests whether the description
        // is a region or locset or iexpr, and updates the label dictionary appropriately.
        // Errors occur when:
        //  * a region is described with a name that matches an existing locset or iexpr
        //    (and vice versa.)
        //  * the description is not well formed, e.g. it contains a syntax error.
        //  * the description is well-formed, but describes neither a region or locset or iexpr.
        try {
            // Evaluate the s-expression to build a region/locset/iexpr.
            auto result = arborio::parse_label_expression(desc);
            if (!result) { // an error parsing / evaluating description.
                throw result.error();
            }
            else if (result->type() == typeid(arb::region)) { // describes a region.
                dict.set(name, std::move(std::any_cast<arb::region&>(*result)));
                regions[name] = desc;
            }
            else if (result->type() == typeid(arb::locset)) { // describes a locset.
                dict.set(name, std::move(std::any_cast<arb::locset&>(*result)));
                locsets[name] = desc;
            }
            else if (result->type() == typeid(arb::iexpr)) { // describes a iexpr.
                dict.set(name, std::move(std::any_cast<arb::iexpr&>(*result)));
                iexpressions[name] = desc;
            }
            else {
                // Successfully parsed an expression that is neither region nor locset.
                throw util::pprintf("The definition of '{} = {}' does not define a valid region or locset.", name, desc);
            }
        }
        catch (std::string msg) {
            const char* base = "\nError adding the label '{}' = '{}'\n{}\n";
            throw std::runtime_error(util::pprintf(base, name, desc, msg));
        }
        // Exceptions are thrown in parse or eval if an unexpected error occured.
        catch (std::exception& e) {
            const char* msg =
                "\n----- internal error -------------------------------------------"
                "\nError parsing the label: '{}' = '{}'"
                "\n"
                "\n{}"
                "\n"
                "\nPlease file a bug report with this full error message at:"
                "\n    github.com/arbor-sim/arbor/issues"
                "\n----------------------------------------------------------------";
            throw arb::arbor_internal_error(util::pprintf(msg, name, desc, e.what()));
        }
    }

    std::string to_string() const {
        std::string s;
        s += "(label_dict";
        for (const auto& [k, v]: regions) s += util::pprintf(" (region  \"{}\" {})", k, v);
        for (const auto& [k, v]: locsets) s += util::pprintf(" (locset \"{}\" {})", k, v);
        for (const auto& [k, v]: iexpressions) s += util::pprintf(" (iexpr \"{}\" {})", k, v);
        s += ")";
        return s;
    }

    bool contains(const std::string& name) const {
        return regions.contains(name) || locsets.contains(name) || iexpressions.contains(name);
    }

    std::string getitem(const std::string& name) const {
        if (locsets.contains(name)) return locsets.at(name);
        if (regions.contains(name)) return regions.at(name);
        if (iexpressions.contains(name)) return iexpressions.at(name);
        throw pybind11::key_error(name);
    }
};    
}
