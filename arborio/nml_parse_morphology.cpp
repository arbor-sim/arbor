#include <algorithm>
#include <optional>
#include <numeric>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/stitch.hpp>
#include <arbor/util/expected.hpp>

#include <arborio/neuroml.hpp>

#include "nml_parse_morphology.hpp"
#include "xml.hpp"

using std::optional;
using arb::region;
using arb::util::expected;
using arb::util::unexpected;

using namespace std::literals;

namespace arborio {

// Implementation utility classes:

namespace {
// Box is a container of size 0 or 1.

template <typename X>
struct box {
    optional<X> x;

    X* begin() { return x? &(*x): nullptr; }
    X* end() { return x? &(*x)+1: nullptr; }

    const X* begin() const { return x? &(*x): nullptr; }
    const X* end() const { return x? &(*x)+1: nullptr; }

    box() = default;
    box(const X& x): x(x) {}
    box(X&& x): x(std::move(x)) {}

    std::size_t size() const { return !empty(); }
    bool empty() const { return !x; }
};

// Merge two sorted sequences, discarding duplicate values.

template <typename Input1, typename Input2, typename Output>
void unique_merge(Input1 i1, Input1 end1, Input2 i2, Input2 end2, Output out) {
    while (i1!=end1 || i2!=end2) {
        if (i1!=end1) {
            while (i2!=end2 && *i2==*i1) ++i2;
        }

        if (i2==end2 || (i1!=end1 && !(*i2<*i1))) {
            const auto& v = *i1++;
            *out++ = v;
            while (i1!=end1 && *i1==v) ++i1;
        }
        else {
            const auto& v = *i2++;
            *out++ = v;
            while (i2!=end2 && *i2==v) ++i2;
        }
    }
}

// Return vector of depths; sorting object collection by depth will
// give a topological order.
//
// The functional Inset takes an index into the collection of objects,
// and returns a range or collection of indices to that object's precedessors.
//
// If a cycle is encountered, return detected_cycle{i} where i
// is the index of an item in the cycle.

struct cycle_detected { std::size_t index; };

template <typename Inset>
expected<std::vector<std::size_t>, cycle_detected> topological_sort(std::size_t n, Inset inset) {
    using std::begin;
    using std::end;

    constexpr std::size_t unknown = -1;
    constexpr std::size_t cycle = -2;

    std::vector<std::size_t> depth(n, unknown);
    std::stack<std::size_t> stack;

    for (std::size_t i = 0; i<n; ++i) {
        if (depth[i]!=unknown) continue;

        depth[i] = cycle;
        stack.push(i);

        while (!stack.empty()) {
            std::size_t j = stack.top();
            std::size_t d = 0;
            bool resolve = true;

            auto in = inset(j);
            for (auto k = begin(in); k!=end(in); ++k) {
                switch (depth[*k]) {
                case cycle:
                    return unexpected(cycle_detected{*k});
                case unknown:
                    depth[*k] = cycle;
                    stack.push(*k);
                    resolve = false;
                    break;
                default:
                    d = std::max(d, 1+depth[*k]);
                }
            }

            if (resolve) {
                depth[j] = d;
                stack.pop();
            }
        }
    }

    return depth;
}

} // namespace


// Internal representations of NeuroML segment and segmentGroup data:

struct neuroml_segment {
    // Morhpological data:
    non_negative id;
    std::string name;
    optional<arb::mpoint> proximal;
    arb::mpoint distal;
    optional<non_negative> parent_id;
    double along = 1;
    bool spherical = false;
    // Topological depth:
    std::size_t tdepth = 0;
};

struct neuroml_segment_group_subtree {
    // Interval determined by segment ids.
    // Represents both `<path>` and `<subTree>` elements.
    optional<non_negative> from, to;
};

struct neuroml_segment_group_info {
    std::string id;
    std::vector<non_negative> segments;
    std::vector<std::string> includes;
    std::vector<neuroml_segment_group_subtree> subtrees;
};

// Processing of parsed segment/segmentGroup data:

struct neuroml_segment_tree {
    // Segments in topological order:
    auto begin() const { return segments_.begin(); }
    auto end() const { return segments_.end(); }

    // How many segments?
    std::size_t size() const { return segments_.size(); }
    bool empty() const { return segments_.empty(); }

    // Segment by id:
    const neuroml_segment operator[](non_negative id) const {
        return segments_.at(index_.at(id));
    }

    // Children of segment with id.
    const std::vector<non_negative>& children(non_negative id) const {
        static std::vector<non_negative> none{};
        auto iter = children_.find(id);
        return iter!=children_.end()? iter->second: none;
    }

    // Does segment id exist?
    bool contains(non_negative id) const {
        return index_.count(id);
    }

    // Construct from vector of segments. Will happily throw if something doesn't add up.
    explicit neuroml_segment_tree(std::vector<neuroml_segment> segs):
        segments_(std::move(segs))
    {
        if (segments_.empty()) return;
        const std::size_t n_seg = segments_.size();

        // Build index, throw on duplicate id.
        for (std::size_t i = 0; i<n_seg; ++i) {
            if (!index_.insert({segments_[i].id, i}).second) {
                throw nml_bad_segment(segments_[i].id);
            }
        }

        // Check parent relationship is sound.
        for (const auto& s: segments_) {
            if (s.parent_id && !index_.count(*s.parent_id)) {
                throw nml_bad_segment(s.id); // No such parent id.
            }
        }

        // Perform topological sort.
        auto inset = [this](std::size_t i) {
            auto& s = segments_[i];
            return s.parent_id? box{index_.at(*s.parent_id)}: box<std::size_t>{};
        };
        if (auto depths = topological_sort(n_seg, inset)) {
            const auto& d = depths.value();
            for (std::size_t i = 0; i<n_seg; ++i) {
                segments_[i].tdepth = d[i];
            }
        }
        else {
            const auto& seg = segments_[depths.error().index];
            throw nml_cyclic_dependency(std::to_string(seg.id));
        }
        std::sort(segments_.begin(), segments_.end(), [](auto& a, auto& b) { return a.tdepth<b.tdepth; });

        // Check for multiple roots:
        if (n_seg>1 && segments_[1].tdepth==0) throw nml_bad_segment(segments_[1].id);

        // Update index:
        for (std::size_t i = 0; i<n_seg; ++i) {
            index_.at(segments_[i].id) = i;
        }

        // Build child tree:
        for (const auto& seg: segments_) {
            if (seg.parent_id) {
                children_[*seg.parent_id].push_back(seg.id);
            }
        }
    }

private:
    std::vector<neuroml_segment> segments_;
    std::unordered_map<non_negative, std::size_t> index_;
    std::unordered_map<non_negative, std::vector<non_negative>> children_;
};

static std::unordered_map<std::string, std::vector<non_negative>> evaluate_segment_groups(
    std::vector<neuroml_segment_group_info> groups,
    const neuroml_segment_tree& segtree)
{
    const std::size_t n_group = groups.size();

    // Expand subTree/path specifications:
    for (auto& g: groups) {
        try {
            for (auto& subtree: g.subtrees) {
                if (!subtree.from && !subtree.to) {
                    // Matches all segments:
                    for (auto& seg: segtree) {
                        g.segments.push_back(seg.id);
                    }
                }
                else if (!subtree.to) {
                    // Add 'from' and all of its descendents.
                    std::stack<non_negative> pending;
                    pending.push(*subtree.from);

                    while (!pending.empty()) {
                        auto top = pending.top();
                        pending.pop();

                        g.segments.push_back(top);
                        for (auto id: segtree.children(top)) {
                            pending.push(id);
                        }
                    }
                }
                else {
                    // Note: if from is not an ancestor of to, the path is regarded as empty.
                    std::vector<non_negative> path;
                    auto opt_id = subtree.to;
                    for (; opt_id && opt_id!=subtree.from; opt_id = segtree[*opt_id].parent_id) {
                        path.push_back(*opt_id);
                    }
                    if (opt_id==subtree.from) {
                        if (subtree.from) g.segments.push_back(*subtree.from);
                        g.segments.insert(g.segments.end(), path.begin(), path.end());
                    }
                }
            }
        }
        catch (...) {
            throw nml_bad_segment_group(g.id);
        }
    }

    // Build index, throw on duplicate id.
    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t i = 0; i<n_group; ++i) {
        if (!index.insert({groups[i].id, i}).second) {
            throw nml_bad_segment_group(groups[i].id);
        }
    }

    // Build group index -> indices of included groups map.
    std::vector<std::vector<std::size_t>> index_to_included_indices(n_group);
    for (std::size_t i = 0; i<n_group; ++i) {
        const auto& includes = groups[i].includes;
        index_to_included_indices[i].reserve(includes.size());
        for (auto& id: includes) {
            if (!index.count(id)) throw nml_bad_segment_group(groups[i].id);
            index_to_included_indices[i].push_back(index.at(id));
        }
    }

    // Get topological order wrt include relationship.
    std::vector<std::size_t> topo_order(n_group);
    if (auto depths = topological_sort(n_group, [&](auto i) { return index_to_included_indices[i]; })) {
        const auto& d = depths.value();
        std::iota(topo_order.begin(), topo_order.end(), std::size_t(0));
        std::sort(topo_order.begin(), topo_order.end(), [&d](auto& a, auto& b) { return d[a]<d[b]; });
    }
    else {
        const auto& group = groups[depths.error().index];
        throw nml_cyclic_dependency(group.id);
    }

    // Accumulate included group segments, following topological order.
    for (auto i: topo_order) {
        auto& g = groups[i];
        std::sort(g.segments.begin(), g.segments.end());

        if (index_to_included_indices[i].empty()) {
            g.segments.erase(std::unique(g.segments.begin(), g.segments.end()), g.segments.end());
        }
        else {
            std::vector<non_negative> merged;
            for (auto j: index_to_included_indices[i]) {
                merged.clear();
                unique_merge(g.segments.begin(), g.segments.end(),
                    groups[j].segments.begin(), groups[j].segments.end(), std::back_inserter(merged));
                std::swap(g.segments, merged);
            }
        }
    }

    // Move final segment lists into map.
    std::unordered_map<std::string, std::vector<non_negative>> group_seg_map;
    for (auto& g: groups) {
        group_seg_map[g.id] = std::move(g.segments);
    }

    return group_seg_map;
}

static arb::stitched_morphology construct_morphology(const neuroml_segment_tree& segtree) {
    arb::stitch_builder builder;
    if (segtree.empty()) return arb::stitched_morphology{builder};

    // Construct result from topologically sorted segments.

    for (const auto& s: segtree) {
        arb::mstitch stitch(std::to_string(s.id), s.distal);
        double along = s.along;

        if (s.spherical) {
            arb_assert(!s.parent_id); // root segment only!
            arb_assert(s.proximal && s.proximal.value()==s.distal);

            arb::mpoint centre = s.distal;
            double radius = centre.radius;

            stitch.prox = arb::mpoint{centre.x, centre.y-radius, centre.z, radius};
            stitch.dist = arb::mpoint{centre.x, centre.y+radius, centre.z, radius};
        }
        else if (s.parent_id && segtree[s.parent_id.value()].spherical) {
            // Check if _parent_ is spherical. If so, we must attach to its centre.
            along = 0.5;
        }
        else {
            stitch.prox = s.proximal;
        }

        if (s.parent_id) {
            builder.add(stitch, std::to_string(s.parent_id.value()), along);
        }
        else {
            builder.add(stitch);
        }
    }

    return arb::stitched_morphology(std::move(builder));
}

nml_morphology_data nml_parse_morphology_element(const xml_node& morph,
                                                 enum neuroml_options::values options) {
    using namespace neuroml_options;
    nml_morphology_data M;
    M.id = get_attr<std::string>(morph, "id");

    std::vector<neuroml_segment> segments;

    const char* q_parent = "./parent";
    const char* q_proximal = "./proximal";
    const char* q_distal = "./distal";

    for (auto xn: morph.select_nodes("./segment")) {
        auto n = xn.node();
        neuroml_segment seg;
        try {
            seg.id = get_attr<unsigned>(n, "id");
            auto name = get_attr<std::string>(n, "name", "");
            auto parent = n.select_node(q_parent).node();
            if (!parent.empty()) {
                seg.parent_id = get_attr<unsigned>(parent, "segment");
                seg.along = get_attr<double>(parent, "fractionAlong", 1.0);
            }

            auto prox = n.select_node(q_proximal).node();
            if (!prox.empty()) {
                double x = get_attr<double>(prox, "x");
                double y = get_attr<double>(prox, "y");
                double z = get_attr<double>(prox, "z");
                double diameter = get_attr<double>(prox, "diameter");
                if (diameter<0) throw nml_bad_segment(seg.id);
                seg.proximal = arb::mpoint{x, y, z, diameter/2};
            }

            if (!seg.parent_id && !seg.proximal) throw nml_bad_segment(seg.id);

            auto dist = n.select_node(q_distal).node();
            if (!dist.empty()) {
                double x = get_attr<double>(dist, "x");
                double y = get_attr<double>(dist, "y");
                double z = get_attr<double>(dist, "z");
                double diameter = get_attr<double>(dist, "diameter");
                if (diameter<0) throw nml_bad_segment(seg.id);
                seg.distal = arb::mpoint{x, y, z, diameter/2};

                // Set spherical flag if we have no parent, options has allow_spherical_root flag,
                // and proximal == distal.
                seg.spherical = (options & allow_spherical_root) && !seg.parent_id && seg.proximal && seg.proximal.value()==seg.distal;
            }
            else {
                throw nml_bad_segment(seg.id);
            }
        }
        catch (nml_parse_error& e) {
            throw nml_bad_segment(seg.id);
        }
        segments.push_back(std::move(seg));
    }

    if (segments.empty()) return M;

    // Compute tree now to save further parsing if something goes wrong.
    neuroml_segment_tree segtree(std::move(segments));

    const char* q_member = "./member";
    const char* q_include = "./include";
    const char* q_path = "./path";
    const char* q_from = "./from";
    const char* q_to = "./to";
    const char* q_subtree = "./subTree";

    std::vector<neuroml_segment_group_info> groups;

    for (auto xn: morph.select_nodes("./segmentGroup")) {
        auto n = xn.node();
        neuroml_segment_group_info group;
        try {
            group.id = get_attr<std::string>(n, "id");
            for (auto xelem: n.select_nodes(q_member)) {
                auto elem = xelem.node();
                auto seg_id = get_attr<unsigned>(elem, "segment");;
                if (!segtree.contains(seg_id)) throw nml_bad_segment_group(group.id);
                group.segments.push_back(get_attr<unsigned>(elem, "segment"));
            }
            for (auto xelem: n.select_nodes(q_include)) {
                auto elem = xelem.node();
                group.includes.push_back(get_attr<std::string>(elem, "segmentGroup"));
            }

            // Treat `<path>` and `<subTree>` identically:
            auto parse_subtree_elem = [&](const auto& elem) {
                auto froms = elem.select_node(q_from).node();
                auto tos = elem.select_node(q_to).node();

                neuroml_segment_group_subtree sub;
                if (!froms.empty()) sub.from = get_attr<unsigned>(froms, "segment");
                if (!tos.empty()) sub.to = get_attr<unsigned>(tos, "segment");
                return sub;
            };

            for (auto elem: n.select_nodes(q_path)) {
                group.subtrees.push_back(parse_subtree_elem(elem.node()));
            }
            for (auto elem: n.select_nodes(q_subtree)) {
                group.subtrees.push_back(parse_subtree_elem(elem.node()));
            }
        }
        catch (nml_parse_error& e) {
            throw nml_bad_segment_group(group.id);
        }

        groups.push_back(std::move(group));
    }

    M.group_segments = evaluate_segment_groups(std::move(groups), segtree);

    // Build morphology and label dictionaries:

    arb::stitched_morphology stitched = construct_morphology(segtree);
    M.morphology = stitched.morphology();
    M.segments = stitched.labels();

    std::unordered_multimap<std::string, non_negative> name_to_ids;
    std::unordered_set<std::string> names;

    for (auto& s: segments) {
        if (!s.name.empty()) {
            name_to_ids.insert({s.name, s.id});
            names.insert(s.name);
        }
    }

    for (const auto& name: names) {
        arb::region r;
        auto ids = name_to_ids.equal_range(name);
        for (auto i = ids.first; i!=ids.second; ++i) {
            r = join(std::move(r), M.segments.regions().at(std::to_string(i->second)));
        }
        M.named_segments.set(name, std::move(r));
    }

    for (const auto& [group_id, segment_ids]: M.group_segments) {
        arb::region r;
        for (auto id: segment_ids) {
            r = join(std::move(r), M.segments.regions().at(std::to_string(id)));
        }
        M.groups.set(group_id, std::move(r));
    }

    return M;
}

} // namespace arborio
