#include <set>
#include <numeric>

#include <arborio/swcio.hpp>

namespace arb {

swc_no_soma::swc_no_soma(int record_id):
    swc_error("No soma found at the root", record_id)
{}

swc_non_consecutive_soma::swc_non_consecutive_soma (int record_id):
    swc_error("Soma samples (tag 1) are not listed consecutively", record_id)
{}

swc_non_serial_soma::swc_non_serial_soma (int record_id):
    swc_error("Soma samples (tag 1) are not listed serially", record_id)
{}

swc_branchy_soma::swc_branchy_soma (int record_id):
    swc_error("Non-soma sample (tag >= 1) connected to a non-distal sample of the soma", record_id)
{}

swc_collocated_soma::swc_collocated_soma(int record_id) :
    swc_error("The samples that make the soma (tag 1) are not allowed to be collocated", record_id)
{}

swc_single_sample_segment::swc_single_sample_segment(int record_id) :
    swc_error("Segments connected to the soma (tag 1) must have 2 samples with the same tag", record_id)
{}

arb::segment_tree load_swc_neuron(const std::vector<swc_record>& records) {
    if (records.empty()) return {};

    auto soma_prox = records.front();

    // Assert that root sample has tag 1.
    if (soma_prox.tag != 1) {
        throw swc_no_soma{soma_prox.id};
    }

    // check for single soma cell
    bool has_children = false;

    // Map of SWC record id to index in `records`.
    std::unordered_map<int, std::size_t> record_index;
    record_index[soma_prox.id] = 0;

    // Vector of records that make up the soma
    std::vector<swc_record> soma_records = {soma_prox};
    int prev_tag = soma_prox.tag;
    int prev_id = soma_prox.id;

    for (std::size_t i = 1; i < records.size(); ++i) {
        const auto& r = records[i];
        record_index[r.id] = i;

        if (r.tag == soma_prox.tag && prev_tag != soma_prox.tag) {
            throw swc_non_consecutive_soma{r.id};
        }

        if (r.tag == soma_prox.tag) {
            if (r.parent_id != prev_id) {
                throw swc_non_serial_soma{r.id};
            }
            soma_records.push_back(r);
        }

        // Find record index of the parent
        auto iter = record_index.find(r.parent_id);
        if (iter == record_index.end() || records[iter->second].id == r.id) throw bad_swc_data{r.id};
        auto parent_record = records[iter->second];

        if (r.tag != 1 && parent_record.tag == 1 && r.parent_id != soma_records.back().id) {
            throw swc_branchy_soma{r.id};
        }

        if (r.tag != 1 && parent_record.tag == 1) {
            has_children = true;
        }

        prev_tag = r.tag;
        prev_id = r.id;
    }

    segment_tree tree;
    tree.reserve(records.size());

    // Map of SWC record id to index in `tree`.
    std::unordered_map<int, msize_t> tree_index;

    // First, construct the soma
    if (soma_records.size() == 1) {
        if (!has_children) {
            // Model the soma as a 1 cylinder with total length=2*radius, extended along the y axis
            tree.append(mnpos, {soma_prox.x, soma_prox.y - soma_prox.r, soma_prox.z, soma_prox.r},
                        {soma_prox.x, soma_prox.y + soma_prox.r, soma_prox.z, soma_prox.r}, 1);
            return tree;
        } else {
            // Model the soma as a 2 cylinders with total length=2*radius, extended along the y axis
            auto p = tree.append(mnpos, {soma_prox.x, soma_prox.y - soma_prox.r, soma_prox.z, soma_prox.r},
                                 {soma_prox.x, soma_prox.y, soma_prox.z, soma_prox.r}, 1);
            tree.append(p, {soma_prox.x, soma_prox.y, soma_prox.z, soma_prox.r},
                        {soma_prox.x, soma_prox.y + soma_prox.r, soma_prox.z, soma_prox.r}, 1);
            tree_index[soma_prox.id] = p;
        }
    } else {
        if (!has_children) {
            // Don't split soma at the midpoint
            msize_t parent = mnpos;
            bool collocated_samples = true;
            for (std::size_t i = 0; i < soma_records.size() - 1; ++i) {
                const auto& p0 = soma_records[i];
                const auto& p1 = soma_records[i + 1];
                parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
                collocated_samples &= ((p0.x == p1.x) && (p0.y == p1.y) && (p0.z == p1.z));
            }
            if (collocated_samples) {
                throw swc_collocated_soma{records[0].id};
            }
            return tree;
        } else {
            // Calculate segment lengths
            bool collocated_samples = true;
            std::vector<double> soma_segment_lengths;
            for (std::size_t i = 1; i < soma_records.size(); ++i) {
                const auto& p0 = soma_records[i - 1];
                const auto& p1 = soma_records[i];
                soma_segment_lengths.push_back(
                        distance(mpoint{p0.x, p0.y, p0.z, p0.r}, mpoint{p1.x, p1.y, p1.z, p1.r}));
                collocated_samples &= ((p0.x == p1.x) && (p0.y == p1.y) && (p0.z == p1.z));
            }
            if (collocated_samples) {
                throw swc_collocated_soma{records[0].id};
            }
            double midlength = std::accumulate(soma_segment_lengths.begin(), soma_segment_lengths.end(), 0.) / 2;

            std::size_t idx = 0;
            for (; idx < soma_segment_lengths.size(); ++idx) {
                auto l = soma_segment_lengths[idx];
                if (midlength > l) {
                    midlength -= l;
                    continue;
                }
                break;
            }

            // Interpolate along the segment that contains the midpoint of the soma
            double pos_on_segment = midlength / soma_segment_lengths[idx];

            auto& r0 = soma_records[idx];
            auto& r1 = soma_records[idx + 1];

            auto x = r0.x + pos_on_segment * (r1.x - r0.x);
            auto y = r0.y + pos_on_segment * (r1.y - r0.y);
            auto z = r0.z + pos_on_segment * (r1.z - r0.z);
            auto r = r0.r + pos_on_segment * (r1.r - r0.r);

            mpoint mid_soma = {x, y, z, r};

            // Construct the soma
            msize_t parent = mnpos;
            for (std::size_t i = 0; i < idx; ++i) {
                const auto& p0 = soma_records[i];
                const auto& p1 = soma_records[i + 1];
                parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
            }
            auto soma_seg = tree.append(parent, {r0.x, r0.y, r0.z, r0.r}, mid_soma, 1);

            if (mpoint r1_p = {r1.x, r1.y, r1.z, r1.r}; mid_soma != r1_p) {
                parent = tree.append(soma_seg, mid_soma, r1_p, 1);
            } else {
                parent = soma_seg;
            }

            for (std::size_t i = idx + 1; i < soma_records.size() - 1; ++i) {
                const auto& p0 = soma_records[i];
                const auto& p1 = soma_records[i + 1];
                parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
            }

            tree_index[soma_records.back().id] = soma_seg;
        }
    }

    // Build branches off soma.
    std::set<int> unused_samples;
    for (const auto& r: records) {
        // Skip the soma samples
        if (r.tag == 1) continue;

        const auto p = r.parent_id;

        // Find parent segment of the record
        auto pseg_iter = tree_index.find(p);
        if (pseg_iter == tree_index.end()) throw bad_swc_data{r.id};

        // Find parent record of the record
        auto prec_iter = record_index.find(p);
        if (prec_iter == record_index.end() || records[prec_iter->second].id == r.id) throw bad_swc_data{r.id};

        // If the sample has a soma sample as its parent don't create a segment.
        if (records[prec_iter->second].tag == 1) {
            // Map the sample id to the segment id of the soma (parent)
            tree_index[r.id] = pseg_iter->second;
            unused_samples.insert(r.id);
            continue;
        }

        const auto& prox = records[prec_iter->second];
        tree_index[r.id] = tree.append(pseg_iter->second, {prox.x, prox.y, prox.z, prox.r}, {r.x, r.y, r.z, r.r},
                                       r.tag);
        unused_samples.erase(prox.id);
    }

    if (!unused_samples.empty()) {
        throw swc_single_sample_segment(*unused_samples.begin());
    }
    return tree;
}
} //namespace arb