#include <arborio/debug.hpp>

#include <arbor/morph/primitives.hpp>

#include <map>
#include <numeric>

namespace arborio {

template <typename T, typename P>
std::vector<std::string> render(const T& tree,
                                arb::msize_t root,
                                const std::multimap<arb::msize_t, arb::msize_t>& children,
                                P print) {
    // ASCII art elements
    // TODO these could be customizable, but need conformant lengths
    const std::string vline = " | ";
    const std::string hline = "---";
    const std::string blank = "   ";
    const std::string split = "-+-";
    const std::string start = " +-";

    auto n_child = children.count(root);
    auto seg = print(root, tree);
    if (0 == n_child) return {seg};

    auto sep = std::string(seg.size(), ' ');
    const auto& [beg, end] = children.equal_range(root);

    std::vector res = {seg};
    arb::msize_t cdx = 0;
    for (auto it = beg; it != end; ++it) {
        const auto& [parent, child] = *it;
        auto rows = render(tree, child, children, print);
        auto rdx = 0;
        for (const auto& row: rows) {
            // Append the first row directly onto our segments, this [- -] -- [- -]
            if (rdx == 0) {
                // The first child of a node may span a sub-tree
                if (cdx == 0) {
                    res.back() += split + row;
                } else {
                    // Other children get connected to the vertical line
                    res.push_back(sep + start + row);
                }
                cdx++;
            } else {
                // If there are more children, extend the subtree by showing a
                // vertical line
                res.push_back(sep + (cdx < n_child ? vline : blank) + row);
            }
            ++rdx;
        }
    }
    // res.push_back(sep);
    return res;
}

ARB_ARBORIO_API std::string default_segment_printer(const arb::msize_t id, const arb::segment_tree&) {
    auto lbl = (id == arb::mnpos) ? "(root)" : std::to_string(id);
    return "[-- id=" + lbl + " --]" ;
}

std::string ARB_ARBORIO_API default_branch_printer(const arb::msize_t id, const arb::morphology& mrf) {
    auto lbl = (id == arb::mnpos) ? std::string("(root)") : std::to_string(id);
    return "<-- id=" + std::to_string(id) + " len=" + std::to_string(mrf.branch_segments(id).size()) + " -->" ;
}

ARB_ARBORIO_API std::string show(const arb::segment_tree& tree) {
    if (tree.empty()) return "";

    std::multimap<arb::msize_t, arb::msize_t> children;
    const auto& ps = tree.parents();
    for (arb::msize_t idx = 0; idx < tree.size(); ++idx) {
        auto parent = ps[idx];
        children.emplace(parent, idx);
    }

    auto res = render(tree, 0, children, default_segment_printer);
    return std::accumulate(res.begin(), res.end(),
                           std::string{},
                           [](auto lhs, auto rhs) { return lhs + rhs + "\n"; });
}

ARB_ARBORIO_API std::string show(const arb::morphology& mrf) {
    if (mrf.empty()) return "";

    std::multimap<arb::msize_t, arb::msize_t> children;
    for (arb::msize_t idx = 0; idx < mrf.num_branches(); ++idx) {
        auto parent = mrf.branch_parent(idx);
        children.emplace(parent, idx);
    }

    auto res = render(mrf, 0, children, default_branch_printer);
    return std::accumulate(res.begin(), res.end(),
                           std::string{},
                           [](auto lhs, auto rhs) { return lhs + rhs + "\n"; });
}
}
