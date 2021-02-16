#include <iostream>
#include <fstream>

#include <arborio/neurolucida.hpp>

#include <optional>
#include <s_expr.hpp>

namespace arborio {

asc_no_document::asc_no_document():
    asc_exception("no neurolucida asc file to parse")
{}

std::optional<std::string_view> head_symbol(const arb::s_expr& e) {
    auto b = e.begin();
    if (b->is_atom()) {
        if (auto& a = b->atom(); a.kind==arb::tok::symbol) {
            return a.spelling;
        }
    }

    return std::nullopt;
}

std::optional<const arb::s_expr*>  match(std::string_view name, const arb::s_expr& e) {
    if (auto s = head_symbol(e); *s==name) {
        return &e.tail();
    }

    return std::nullopt;
}

asc_morphology load_asc(std::string filename) {
    std::ifstream fid(filename);

    std::cout << "loading " << filename << "\n";
    if (!fid.good()) {
        throw asc_no_document();
    }

    std::string fstr; // will hold contents of input file
    fid.seekg(0, std::ios::end);
    fstr.reserve(fid.tellg());
    fid.seekg(0, std::ios::beg);

    fstr.assign((std::istreambuf_iterator<char>(fid)),
                 std::istreambuf_iterator<char>());

    arb::transmogrifier stream(fstr, {{',', " "},
                                     {'|', ")("},
                                     {'<', "(spine "},
                                     {'>', ")"}});

    auto parsed = arb::parse_multi_s_expr(stream);

    std::cout << "parsed with " << parsed.size() << " fields\n";

    auto top_level_expr = std::begin(parsed);

    while (top_level_expr!=std::end(parsed)) {
        if (auto args = match("ImageCoords", *top_level_expr)) {
            // Do nothing.
            // Might contain meta data for viewing/rendering in the Neurolucida GUI.
            std::cout << "#### ImageCoords: " << **args << "\n";
        }
        else if (auto args = match("Sections", *top_level_expr)) {
            // Do nothing: this also has optional arguments that don't seem to
            // impact the geometry itself.
            std::cout << "#### Sections: " << **args << "\n";
        }
        ++top_level_expr;
    }

    std::cout << "\n------------------------\n\n";

    return {};
}

} // namespace arborio
