#include <cstring>
#include <ostream>
#include <fstream>
#include <numeric>

#include <arbor/util/expected.hpp>

#include <arborio/neurolucida.hpp>
#include "arbor/arbexcept.hpp"
#include "arbor/morph/primitives.hpp"
#include "asc_lexer.hpp"

#include <optional>

namespace arborio {

asc_parse_error::asc_parse_error(const std::string& error_msg, unsigned line, unsigned column):
    asc_exception("asc parser error (line "+std::to_string(line)+" col "+std::to_string(column)+"): "+error_msg),
    message(error_msg),
    line(line),
    column(column)
{}

asc_unsupported::asc_unsupported(const std::string& error_msg):
    asc_exception("unsupported in asc description: "+error_msg),
    message(error_msg)
{}


namespace {
// Parse functions and internal representations kept in unnamed namespace.

struct parse_error {
    struct cpp_info {
        const char* file;
        int line;
    };

    std::string msg;
    asc::src_location loc;
    std::vector<cpp_info> stack;

    parse_error(std::string m, asc::src_location l, cpp_info cpp):
        msg(std::move(m)), loc(l)
    {
        stack.push_back(cpp);
    }

    parse_error& append(const cpp_info& i) {
        stack.emplace_back(i);
        return *this;
    }

    asc_parse_error make_exception() const {
        // Uncomment to print a stack trace of the parser.
        // A very useful tool for investigating invalid inputs or parser bugs.

        //for (auto& frame: stack) std::cout << "  " << frame.file << ":" << frame.line << "\n";

        return {msg, loc.line, loc.column};
    }
};

template <typename T>
using parse_hopefully = arb::util::expected<T, parse_error>;
using arb::util::unexpected;
using asc::tok;

#define PARSE_ERROR(msg, loc) parse_error(msg, loc, {__FILE__, __LINE__})
#define FORWARD_PARSE_ERROR(err) arb::util::unexpected(parse_error(std::move(err).append({__FILE__, __LINE__})))

// The parse_* functions will attempt to parse an expected token from the lexer.

// Attempt to parse a token of the expected kind. On succes the token is returned,
// otherwise a parse_error.
parse_hopefully<tok> expect_token(asc::lexer& l, tok kind) {
    auto& t = l.current();
    if (t.kind != kind) {
        return unexpected(PARSE_ERROR("unexpected symbol '"+t.spelling+"'", t.loc));
    }
    l.next();
    return kind;
}

#define EXPECT_TOKEN(L, TOK) {if (auto rval__ = expect_token(L, TOK); !rval__) return FORWARD_PARSE_ERROR(rval__.error());}

// Attempt to parse a double precision value from the input stream.
// Will consume both integer and real values.
parse_hopefully<double> parse_double(asc::lexer& L) {
    auto t = L.current();
    if (!(t.kind==tok::integer || t.kind==tok::real)) {
        return unexpected(PARSE_ERROR("missing real number", L.current().loc));
    }
    L.next(); // consume the number
    return std::stod(t.spelling);
}

#define PARSE_DOUBLE(L, X) {if (auto rval__ = parse_double(L)) X=*rval__; else return FORWARD_PARSE_ERROR(rval__.error());}

// Attempt to parse an integer in the range [0, 256).
parse_hopefully<std::uint8_t> parse_uint8(asc::lexer& L) {
    auto t = L.current();
    if (t.kind!=tok::integer) {
        return unexpected(PARSE_ERROR("missing uint8 number", L.current().loc));
    }

    // convert to large integer and test
    auto value = std::stoll(t.spelling);
    if (value<0 || value>255) {
        return unexpected(PARSE_ERROR("value out of range [0, 255]", L.current().loc));
    }
    L.next(); // consume token
    return static_cast<std::uint8_t>(value);
}

#define PARSE_UINT8(L, X) {if (auto rval__ = parse_uint8(L)) X=*rval__; else return FORWARD_PARSE_ERROR(rval__.error());}

// Find the matching closing parenthesis, and consume it.
// Assumes that opening paren has been consumed.
void parse_to_closing_paren(asc::lexer& L, unsigned depth=0) {
    while (true) {
        const auto& t = L.current();
        switch (t.kind) {
            case tok::lparen:
                L.next();
                ++depth;
                break;
            case tok::rparen:
                L.next();
                if (depth==0) return;
                --depth;
                break;
            case tok::error:
                throw asc_parse_error(t.spelling, t.loc.line, t.loc.column);
            case tok::eof:
                throw asc_parse_error("unexpected end of file", t.loc.line, t.loc.column);
            default:
                L.next();
        }
    }
}

bool parse_if_symbol_matches(const char* match, asc::lexer& L) {
    auto& t = L.current();
    if (t.kind==tok::symbol && !std::strcmp(match, t.spelling.c_str())) {
        L.next();
        return true;
    }
    return false;
}

bool symbol_matches(const char* match, const asc::token& t) {
    return t.kind==tok::symbol && !std::strcmp(match, t.spelling.c_str());
}

// A list of symbols that indicate markers
bool is_marker_symbol(const asc::token& t) {
    return symbol_matches("Dot", t)
        || symbol_matches("OpenCircle", t)
        || symbol_matches("Cross", t);
};


// Parse a color expression, which have been observed in the wild in two forms:
//  (Color Red)                 ; labeled
//  (Color RGB (152, 251, 152)) ; RGB literal
struct asc_color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

[[maybe_unused]]
std::ostream& operator<<(std::ostream& o, const asc_color& c) {
    return o << "(asc-color " << (int)c.r << " " << (int)c.g << " " << (int)c.b << ")";
}

std::unordered_map<std::string, asc_color> color_map = {
    {"Black",     {  0,   0,   0}},
    {"White",     {255, 255, 255}},
    {"Red",       {255,   0,   0}},
    {"Lime",      {  0, 255,   0}},
    {"Blue",      {  0,   0, 255}},
    {"Yellow",    {255, 255,   0}},
    {"Cyan",      {  0, 255, 255}},
    {"Aqua",      {  0, 255, 255}},
    {"Magenta",   {255,   0, 255}},
    {"Fuchsia",   {255,   0, 255}},
    {"Silver",    {192, 192, 192}},
    {"Gray",      {128, 128, 128}},
    {"Maroon",    {128,   0,   0}},
    {"Olive",     {128, 128,   0}},
    {"Green",     {  0, 128,   0}},
    {"Purple",    {128,   0, 128}},
    {"Teal",      {  0, 128, 128}},
    {"Navy",      {  0,   0, 128}},
    {"Orange",    {255, 165,   0}},
};

parse_hopefully<asc_color> parse_color(asc::lexer& L) {
    EXPECT_TOKEN(L, tok::lparen);
    if (!symbol_matches("Color", L.current())) {
        return unexpected(PARSE_ERROR("expected Color symbol missing", L.current().loc));
    }
    // consume Color symbol
    auto t = L.next();

    if (parse_if_symbol_matches("RGB", L)) {
        // Read RGB triple in the form (r, g, b)

        EXPECT_TOKEN(L, tok::lparen);

        asc_color color;
        PARSE_UINT8(L, color.r);
        EXPECT_TOKEN(L, tok::comma);
        PARSE_UINT8(L, color.g);
        EXPECT_TOKEN(L, tok::comma);
        PARSE_UINT8(L, color.b);


        EXPECT_TOKEN(L, tok::rparen);
        EXPECT_TOKEN(L, tok::rparen);

        return color;
    }
    else if (t.kind==tok::symbol) {
        // Look up the symbol in the table
        if (auto it = color_map.find(t.spelling); it!=color_map.end()) {
            L.next();
            EXPECT_TOKEN(L, tok::rparen);
            return it->second;
        }
        else {
            return unexpected(PARSE_ERROR("unknown color value '"+t.spelling+"'", t.loc));
        }
    }

    return unexpected(PARSE_ERROR("unexpected symbol in Color description \'"+t.spelling+"\'", t.loc));
}

#define PARSE_COLOR(L, X) {if (auto rval__ = parse_color(L)) X=*rval__; else return FORWARD_PARSE_ERROR(rval__.error());}

// Parse zSmear statement, which has the form:
//  (zSmear alpha beta)
// Where alpha and beta are double precision values.
//
//      Used to alter the Used to alter the displayed thickness of dendrites to
//      resemble the optical aberration in z. This can be caused by both the
//      point spread function and by refractive index mismatch between the specimen
//      and the lens immersion medium. The diameter of a branch in z is adjusted
//      using the following equation, Dz= Dxy*S, where Dxy is the recorded
//      centerline diameter on the xy plane and S is the smear factor. The smear
//      factor is calculated using this equation, S=α*Dxyβ. The minimum diameter
//      is 1.0 µm, even if S values are less than 1.0.
//
// Which doesn't make much sense to track with our segment-based representation.

struct zsmear {
    double alpha;
    double beta;
};

[[maybe_unused]]
std::ostream& operator<<(std::ostream& o, const zsmear& z) {
    return o << "(zsmear " << z.alpha << " " << z.beta << ")";
}

parse_hopefully<zsmear> parse_zsmear(asc::lexer& L) {
    // check and consume opening paren
    EXPECT_TOKEN(L, tok::lparen);

    if (!symbol_matches("zSmear", L.current())) {
        return unexpected(PARSE_ERROR("expected zSmear symbol missing", L.current().loc));
    }
    // consume zSmear symbol
    auto t = L.next();

    zsmear s;
    PARSE_DOUBLE(L, s.alpha);
    PARSE_DOUBLE(L, s.beta);

    // check and consume closing paren
    EXPECT_TOKEN(L, tok::rparen);

    return s;
}

#define PARSE_ZSMEAR(L, X) if (auto rval__ = parse_zsmear(L)) X=*rval__; else return FORWARD_PARSE_ERROR(rval__.error());

parse_hopefully<arb::mpoint> parse_point(asc::lexer& L) {
    // check and consume opening paren
    EXPECT_TOKEN(L, tok::lparen);

    arb::mpoint p;
    PARSE_DOUBLE(L, p.x);
    PARSE_DOUBLE(L, p.y);
    PARSE_DOUBLE(L, p.z);
    double diameter;
    PARSE_DOUBLE(L, diameter);
    p.radius = diameter/2.0;

    // check and consume closing paren
    EXPECT_TOKEN(L, tok::rparen);

    return p;
}

#define PARSE_POINT(L, X) if (auto rval__ = parse_point(L)) X=*rval__; else return FORWARD_PARSE_ERROR(rval__.error());

parse_hopefully<arb::mpoint> parse_spine(asc::lexer& L) {
    EXPECT_TOKEN(L, tok::lt);
    auto& t = L.current();
    while (t.kind!=tok::gt && t.kind!=tok::error && t.kind!=tok::eof) {
        L.next();
    }
    //if (t.kind!=error && t.kind!=eof)
    EXPECT_TOKEN(L, tok::gt);

    return arb::mpoint{};
}

#define PARSE_SPINE(L, X) if (auto rval__ = parse_spine(L)) X=std::move(*rval__); else return FORWARD_PARSE_ERROR(rval__.error());

parse_hopefully<std::string> parse_name(asc::lexer& L) {
    EXPECT_TOKEN(L, tok::lparen);
    if (!symbol_matches("Name", L.current())) {
        return unexpected(PARSE_ERROR("expected Name symbol missing", L.current().loc));
    }

    // consume Name symbol
    auto t = L.next();
    if (t.kind != tok::string) {
        return unexpected(PARSE_ERROR("expected a string in name description", t.loc));
    }
    std::string name = t.spelling;

    L.next();
    EXPECT_TOKEN(L, tok::rparen);

    return name;
}

#define PARSE_NAME(L, X) {if (auto rval__ = parse_name(L)) X=*rval__; else return FORWARD_PARSE_ERROR(rval__.error());}

struct marker_set {
    asc_color color;
    std::string name;
    std::vector<arb::mpoint> locations;
};

[[maybe_unused]]
std::ostream& operator<<(std::ostream& o, const marker_set& ms) {
    o << "(marker-set \"" << ms.name << "\" " << ms.color;
    for (auto& l: ms.locations) o << " " << l;
    return o << ")";

}

parse_hopefully<marker_set> parse_markers(asc::lexer& L) {
    EXPECT_TOKEN(L, tok::lparen);

    marker_set markers;

    // parse marker kind keyword
    auto t = L.current();
    if (!is_marker_symbol(t)) {
        return unexpected(PARSE_ERROR("expected a valid marker type", t.loc));
    }
    L.next();
    while (L.current().kind==tok::lparen) {
        auto n = L.peek();
        if (symbol_matches("Color", n)) {
            PARSE_COLOR(L, markers.color);
        }
        else if (symbol_matches("Name", n)) {
            PARSE_NAME(L, markers.name);
        }
        else {
            arb::mpoint loc;
            PARSE_POINT(L, loc);
            markers.locations.push_back(loc);
        }
    }

    EXPECT_TOKEN(L, tok::rparen);

    return markers;
}

#define PARSE_MARKER(L, X) if (auto rval__ = parse_markers(L)) X=std::move(*rval__); else return FORWARD_PARSE_ERROR(rval__.error());

struct branch {
    std::vector<arb::mpoint> samples;
    std::vector<branch> children;
};

std::size_t num_samples(const branch& b) {
    return b.samples.size() + std::accumulate(b.children.begin(), b.children.end(), std::size_t(0), [](std::size_t x, const branch& b) {return x+num_samples(b);});
}

struct sub_tree {
    constexpr static int no_tag = std::numeric_limits<int>::min();
    std::string name;
    int tag = no_tag;
    branch root;
    asc_color color;
};

// Forward declaration.
parse_hopefully<std::vector<branch>> parse_children(asc::lexer& L);

parse_hopefully<branch> parse_branch(asc::lexer& L) {
    branch B;

    auto& t = L.current();

    // Assumes that the opening parenthesis has already been consumed, because
    // parsing of the first branch in a sub-tree starts on the first sample.

    bool finished = t.kind == tok::rparen;

    auto branch_end = [] (const asc::token& t) {
        return t.kind == tok::pipe || t.kind == tok::rparen;
    };

    // One of these symbols must always be present at what Arbor calls a terminal.
    auto is_branch_end_symbol = [] (const asc::token& t) {
        return symbol_matches("Normal", t)
            || symbol_matches("High", t)
            || symbol_matches("Low", t)
            || symbol_matches("Incomplete", t)
            || symbol_matches("Generated", t);
    };

    // Parse the samples in this branch up to either a terminal or an explicit fork.
    while (!finished) {
        auto p = L.peek();
        // Assume that a sample has been found if the first value after a parenthesis
        // is a number. An error will be returned if that is not the case.
        if (t.kind==tok::lparen && (p.kind==tok::integer || p.kind==tok::real)) {
            arb::mpoint sample;
            PARSE_POINT(L, sample);
            B.samples.push_back(sample);
        }
        // A marker statement is always of the form ( MARKER_TYPE ...)
        else if (t.kind==tok::lparen && is_marker_symbol(p)) {
            marker_set markers;
            PARSE_MARKER(L, markers);
            // Parse the markers, but don't record information about them.
            // These could be grouped into locset by name and added to the label dictionary.
        }
        // Spines are marked by a "less than", i.e. "<", symbol.
        else if (t.kind==tok::lt) {
            arb::mpoint spine;
            PARSE_SPINE(L, spine);
            // parse the spine, but don't record the location.
        }
        // Test for a symbol that indicates a terminal.
        else if (is_branch_end_symbol(t)) {
            L.next(); // Consume symbol
            if (!branch_end(t)) {
                return unexpected(PARSE_ERROR("Incomplete, Normal, High, Low or Generated not at a branch terminal", t.loc));
            }
            finished = true;
        }
        else if (branch_end(t)) {
            finished = true;
        }
        // The end of the branch followed by an explicit fork point.
        else if (t.kind==tok::lparen) {
            finished = true;
        }
        else {
            return unexpected(PARSE_ERROR("Unexpected input '"+t.spelling+"'", t.loc));
        }
    }

    // Recursively parse any child branches.
    if (t.kind==tok::lparen) {
        if (auto kids = parse_children(L)) {
            B.children = std::move(*kids);
        }
        else {
            return FORWARD_PARSE_ERROR(kids.error());
        }
    }

    return B;
}

parse_hopefully<std::vector<branch>> parse_children(asc::lexer& L) {
    std::vector<branch> children;

    auto& t = L.current();

    EXPECT_TOKEN(L, tok::lparen);

    bool finished = t.kind==tok::rparen;
    while (!finished) {
        if (auto b1 = parse_branch(L)) {
            children.push_back(std::move(*b1));
        }
        else {
            return FORWARD_PARSE_ERROR(b1.error());
        }

        // Test for siblings, which are either marked sanely using the obvious
        // logical "(", or with a too-clever-by-half "|".
        finished = !(t.kind==tok::pipe || t.kind==tok::lparen);
        if (!finished) L.next();
    }

    EXPECT_TOKEN(L, tok::rparen);

    return children;
}

parse_hopefully<sub_tree> parse_sub_tree(asc::lexer& L) {
    EXPECT_TOKEN(L, tok::lparen);

    sub_tree tree;

    // Parse the arbitrary, unordered and optional meta-data that may be prepended to the sub-tree.
    //  string label, e.g. "Cell Body"
    //  color, e.g. (Color Red)
    //  z-smear, i.e. (zSmear alpha beta)
    // And the required demarcation of CellBody, Axon, Dendrite or Apical.
    while (true) {
        auto& t = L.current();
        // ASC files have an option to attach a string to the start of a sub-tree.
        // If/when we find out what this string is applied to, we might create a dictionary entry for it.
        if (t.kind == tok::string && !t.spelling.empty()) {
            L.next();
        }
        else if (t.kind == tok::lparen) {
            auto t = L.peek();
            if (symbol_matches("Color", t)) {
                PARSE_COLOR(L, tree.color);
            }
            // Every sub-tree is marked with one of: {CellBody, Axon, Dendrite, Apical}
            // Hence it is possible to assign SWC tags for soma, axon, dend and apic.
            else if (symbol_matches("CellBody", t)) {
                tree.name = t.spelling;
                tree.tag = 1;
                L.next(2); // consume symbol
                EXPECT_TOKEN(L, tok::rparen);
            }
            else if (symbol_matches("Axon", t)) {
                tree.name = t.spelling;
                tree.tag = 2;
                L.next(2); // consume symbol
                EXPECT_TOKEN(L, tok::rparen);
            }
            else if (symbol_matches("Dendrite", t)) {
                tree.name = t.spelling;
                tree.tag = 3;
                L.next(2); // consume symbol
                EXPECT_TOKEN(L, tok::rparen);
            }
            else if (symbol_matches("Apical", t)) {
                tree.name = t.spelling;
                tree.tag = 4;
                L.next(2); // consume symbol
                EXPECT_TOKEN(L, tok::rparen);
            }
            // Ignore zSmear.
            else if (symbol_matches("zSmear", t)) {
                PARSE_ZSMEAR(L, __attribute__((unused)) auto _);
            }
            else if (t.kind==tok::integer || t.kind==tok::real) {
                // Assume that this is the first sample.
                // Break to start parsing the samples in the sub-tree.
                break;
            }
            else {
                return unexpected(PARSE_ERROR("Unexpected input'"+t.spelling+"'", t.loc));
            }
        }
        else if (t.kind == tok::rparen) {
            // The end of the sub-tree expression was reached while parsing the header.
            // Implies that there were no samples in the sub-tree, which we will treat
            // as an error.
            return unexpected(PARSE_ERROR("Empty sub-tree", t.loc));
        }
        else {
            // An unexpected token was encountered.
            return unexpected(PARSE_ERROR("Unexpected input '"+t.spelling+"'", t.loc));
        }
    }

    if (tree.tag==tree.no_tag) {
        return unexpected(PARSE_ERROR("Missing sub-tree label (CellBody, Axon, Dendrite or Apical)", L.current().loc));
    }

    // Now that the meta data has been read, process the samples.
    // parse_branch will recursively construct the sub-tree.
    if (auto branches = parse_branch(L)) {
        tree.root = std::move(*branches);
    }
    else {
        return FORWARD_PARSE_ERROR(branches.error());
    }

    EXPECT_TOKEN(L, tok::rparen);

    return tree;
}

} // namespace


// Perform the parsing of the input as a string.
ARB_ARBORIO_API arb::segment_tree parse_asc_string_raw(const char* input) {
    asc::lexer lexer(input);

    std::vector<sub_tree> sub_trees;

    // Iterate over high-level pseudo-s-expressions in the file.
    // This pass simply parses the contents of the file, to be interpretted
    // in a later pass.
    while (lexer.current().kind != asc::tok::eof) {
        auto t = lexer.current();

        // Test for errors
        if (t.kind == asc::tok::error) {
            throw asc_parse_error(t.spelling, t.loc.line, t.loc.column);
        }

        // Expect that all top-level expressions start with open parenthesis '('
        if (t.kind != asc::tok::lparen) {
            throw asc_parse_error("expect opening '('", t.loc.line, t.loc.column);
        }

        // top level expressions are one of
        //      ImageCoords
        //      Sections
        //      Description
        t = lexer.peek();
        if (symbol_matches("Description", t)) {
            lexer.next();
            parse_to_closing_paren(lexer);
        }
        else if (symbol_matches("ImageCoords", t)) {
            lexer.next();
            parse_to_closing_paren(lexer);
        }
        else if (symbol_matches("Sections", t)) {
            lexer.next();
            parse_to_closing_paren(lexer);
        }
        else {
            if (auto tree = parse_sub_tree(lexer)) {
                sub_trees.push_back(std::move(*tree));
            }
            else {
                throw tree.error().make_exception();
            }
        }
    }


    // Return an empty description if no sub-trees were parsed.
    if (!sub_trees.size()) {
        return {};
    }

    // Process the sub-trees to construct the morphology and labels.
    std::vector<std::size_t> soma_contours;
    for (unsigned i=0; i<sub_trees.size(); ++i) {
        if (sub_trees[i].tag == 1) soma_contours.push_back(i);
    }

    const auto soma_count = soma_contours.size();

    // Assert that there is no more than one CellBody description.
    // This case of multiple contours to define the soma has to be special cased.
    if (soma_count!=1u) {
        throw asc_unsupported("only 1 CellBody contour can be handled");
    }

    arb::segment_tree stree;

    // Form a soma composed of two cylinders, extended along the negative then positive
    // y axis from the center of the soma.
    //
    //          --------  soma_2
    //          |      |
    //          |      |             y
    //          --------  soma_0     |  z
    //          |      |             | /
    //          |      |             |/
    //          --------  soma_1     o---  x
    //

    arb::mpoint soma_0, soma_1, soma_2;
    if (soma_count==1u) {
        const auto& st = sub_trees[soma_contours.front()];
        const auto& samples = st.root.samples;
        if (samples.size()==1u) {
            // The soma is described as a sphere with a single sample.
            soma_0 = samples.front();
        }
        else {
            // The soma is described by a contour.
            const auto ns = samples.size();
            soma_0.x = std::accumulate(samples.begin(), samples.end(), 0., [](double a, auto& s) {return a+s.x;}) / ns;
            soma_0.y = std::accumulate(samples.begin(), samples.end(), 0., [](double a, auto& s) {return a+s.y;}) / ns;
            soma_0.z = std::accumulate(samples.begin(), samples.end(), 0., [](double a, auto& s) {return a+s.z;}) / ns;
            soma_0.radius = std::accumulate(samples.begin(), samples.end(), 0.,
                    [&soma_0](double a, auto& c) {return a+arb::distance(c, soma_0);}) / ns;
        }
        soma_1 = {soma_0.x, soma_0.y-soma_0.radius, soma_0.z, soma_0.radius};
        soma_2 = {soma_0.x, soma_0.y+soma_0.radius, soma_0.z, soma_0.radius};
        stree.append(arb::mnpos, soma_0, soma_1, 1);
        stree.append(arb::mnpos, soma_0, soma_2, 1);
    }

    // Append the dend, axon and apical dendrites.
    for (const auto& st: sub_trees) {
        const int tag = st.tag;

        // Skip soma contours.
        if (tag==1) continue;

        // For now attach everything to the center of the soma at soma_0.
        // Later we could try to attach to whichever end of the soma is closest.
        // Also need to push parent id
        struct binf {
            const branch* child;
            arb::msize_t parent_id;
            arb::mpoint sample;
        };

        std::vector<binf> tails = {{&st.root, arb::mnpos, soma_0}};

        while (!tails.empty()) {
            auto head = tails.back();
            tails.pop_back();

            auto parent = head.parent_id;
            auto& branch = *head.child;
            auto prox_sample = head.sample;

            if (!branch.samples.empty()) { // Skip empty branches, which are permitted
                auto it = branch.samples.begin();
                // Don't connect the first sample to the distal end of the parent
                // branch if the parent is the soma center.
                if (parent==arb::mnpos) {
                    prox_sample = *it;
                    ++it;
                }
                do {
                    parent = stree.append(parent, prox_sample, *it, tag);
                    prox_sample = *it;
                    ++it;
                } while (it!=branch.samples.end());
            }

            // Push child branches to stack in reverse order.
            // This ensures that branches are popped from the stack in the same
            // order they were described in the file, so that segments in the
            // segment tree were added in the same order they are described
            // in the file, to give deterministic branch numbering.
            const auto& kids = branch.children;
            for (auto it=kids.rbegin(); it!=kids.rend(); ++it) {
                tails.push_back({&(*it), parent, prox_sample});
            }
        }
    }

    return stree;
}


ARB_ARBORIO_API asc_morphology parse_asc_string(const char* input) {
    // Parse segment tree
    arb::segment_tree stree = parse_asc_string_raw(input);

    // Construct the morphology.
    arb::morphology morphology(stree);

    // Construct the label dictionary.
    arb::label_dict labels;
    labels.set("soma", arb::reg::tagged(1));
    labels.set("axon", arb::reg::tagged(2));
    labels.set("dend", arb::reg::tagged(3));
    labels.set("apic", arb::reg::tagged(4));

    return {stree, std::move(morphology), std::move(labels)};
}


inline std::string read_file(std::string filename) {
    std::ifstream fid(filename);

    if (!fid.good()) {
        throw arb::file_not_found_error(filename);
    }

    // load contents of the file into a string.
    std::string fstr;
    fid.seekg(0, std::ios::end);
    fstr.reserve(fid.tellg());
    fid.seekg(0, std::ios::beg);

    fstr.assign((std::istreambuf_iterator<char>(fid)),
                 std::istreambuf_iterator<char>());
    return fstr;
}


ARB_ARBORIO_API asc_morphology load_asc(std::string filename) {
    std::string fstr = read_file(filename);
    return parse_asc_string(fstr.c_str());
}


ARB_ARBORIO_API arb::segment_tree load_asc_raw(std::string filename) {
    std::string fstr = read_file(filename);
    return parse_asc_string_raw(fstr.c_str());
}


} // namespace arborio

