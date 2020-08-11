#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <arbor/morph/stitch.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

#include "util/ordered_forest.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"

namespace arb {

struct stitch_builder_impl {
    struct stitch_segment {
        double along_prox;
        double along_dist;

        mpoint prox;
        mpoint dist;
        int tag;
        std::string stitch_id;
        msize_t seg_id;
    };

    using forest_type = util::ordered_forest<stitch_segment>;

    forest_type forest;
    std::unordered_map<std::string, forest_type::iterator> id_to_node;
    std::string last_id;

    stitch_builder_impl() = default;
    stitch_builder_impl(stitch_builder_impl&&) = default;
    stitch_builder_impl(const stitch_builder_impl& other):
        forest(other.forest),
        last_id(other.last_id)
    {
        for (auto i = forest.preorder_begin(); i!=forest.preorder_end(); ++i) {
            id_to_node.insert({i->stitch_id, i});
        }
    }

    stitch_builder_impl& operator=(stitch_builder_impl&&) = default;
    stitch_builder_impl& operator=(const stitch_builder_impl& other) {
        if (this==&other) return *this;
        return *this = stitch_builder_impl(other);
    }

    void add(mstitch f, const std::string& parent, double along) {
        if (id_to_node.count(f.id)) throw duplicate_stitch_id(f.id);

        forest_type::iterator p;

        if (!(parent.empty() && forest.empty())) {
            p = find_stitch_along(parent, along);
            arb_assert(p);

            if (along==p->along_prox) {
                if (!f.prox) f.prox = p->prox;
                p = p.parent();
            }
            else if (along<p->along_dist) {
                // Split parent node p at along.
                auto split = *p;

                mpoint point = lerp(p->prox, p->dist, (along-p->along_prox)/(p->along_dist-p->along_prox));
                if (!f.prox) f.prox = point;

                p->dist = point;
                p->along_dist = along;
                split.prox = point;
                split.along_prox = along;

                auto i = forest.push_child(p, split);
                while (i.next()) {
                    auto tmp = forest.prune_after(i);
                    forest.graft_child(i, std::move(tmp));
                }
            }
            else {
                if (!f.prox) f.prox = p->dist;
            }
        }
        if (!f.prox) throw missing_stitch_start(f.id);

        stitch_segment n{0., 1., f.prox.value(), f.dist, f.tag, f.id, msize_t(-1)};
        id_to_node[f.id] = p? forest.push_child(p, n): forest.push_front(n);
        last_id = f.id;
    }

    forest_type::iterator find_stitch_along(const std::string& id, double along) {
        if (along<0 || along>1) throw invalid_stitch_position(id, along);

        auto map_it = id_to_node.find(id);
        if (map_it==id_to_node.end()) throw no_such_stitch(id);

        auto i = map_it->second;
        arb_assert(i->along_prox==0);
        arb_assert(i->along_dist==1 || i.child());

        while (along>i->along_dist) {
            // Continuation is last child.
            i = i.child();
            arb_assert(i);
            while (i.next()) i = i.next();
        }
        return i;
    }
};

stitch_builder::stitch_builder(): impl_(new stitch_builder_impl) {}

stitch_builder::stitch_builder(stitch_builder&&) = default;
stitch_builder& stitch_builder::operator=(stitch_builder&&) = default;

stitch_builder& stitch_builder::add(mstitch f, const std::string& parent_id, double along) {
    impl_->add(std::move(f), parent_id, along);
    return *this;
}

stitch_builder& stitch_builder::add(mstitch f, double along) {
    return add(std::move(f), impl_->last_id, along);
}

stitch_builder::~stitch_builder() = default;


struct stitched_morphology_impl {
    std::unordered_multimap<std::string, msize_t> id_to_segs;
    segment_tree stree;

    stitched_morphology_impl(stitch_builder_impl bimpl) {
        auto iter = bimpl.forest.preorder_begin();
        auto end = bimpl.forest.preorder_end();

        for (; iter!=end; ++iter) {
            msize_t seg_parent_id = iter.parent()? iter.parent()->seg_id: mnpos;
            iter->seg_id = stree.append(seg_parent_id, iter->prox, iter->dist, iter->tag);
        }

        for (const auto& id_node: bimpl.id_to_node) {
            const std::string& id = id_node.first;
            auto iter = id_node.second;

            while (iter && iter->stitch_id==id) {
                id_to_segs.insert({id, iter->seg_id});
                iter = iter.child();
                while (iter.next()) {
                    iter = iter.next();
                }
            }
        }
    }
};

stitched_morphology::stitched_morphology(stitch_builder&& builder):
    impl_(new stitched_morphology_impl(std::move(*builder.impl_)))
{}

stitched_morphology::stitched_morphology(const stitch_builder& builder):
    impl_(new stitched_morphology_impl(*builder.impl_))
{}

stitched_morphology::stitched_morphology(stitched_morphology&& other) = default;

arb::morphology stitched_morphology::morphology() const {
    return arb::morphology(impl_->stree);
}

label_dict stitched_morphology::labels(const std::string& prefix) const {
    label_dict dict;

    auto i0 = impl_->id_to_segs.begin();
    auto end = impl_->id_to_segs.end();
    while (i0 != end) {
        auto i1 = i0;
        while (i1 != end && i1->first==i0->first) ++i1;

        region r  = util::foldl(
            [&](region r, const auto& elem) { return join(std::move(r), reg::segment(elem.second)); },
            reg::nil(),
            util::make_range(i0, i1));

        dict.set(prefix+i0->first, std::move(r));
        i0 = i1;
    }

    return dict;
}

region stitched_morphology::stitch(const std::string& id) const {
    auto seg_ids = util::make_range(impl_->id_to_segs.equal_range(id));
    if (seg_ids.empty()) throw no_such_stitch(id);

    return util::foldl(
        [&](region r, const auto& elem) { return join(std::move(r), reg::segment(elem.second)); },
        reg::nil(), seg_ids);
}

std::vector<msize_t> stitched_morphology::segments(const std::string& id) const {
    auto seg_ids = util::transform_view(util::make_range(impl_->id_to_segs.equal_range(id)), util::second);
    if (seg_ids.empty()) throw no_such_stitch(id);

    return std::vector<msize_t>(begin(seg_ids), end(seg_ids));
}

stitched_morphology::~stitched_morphology() = default;

} // namespace arb
