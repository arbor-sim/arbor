#include <cmath>

#include <arbor/cable_cell.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/recipe.hpp>

namespace arb {

// Generate a segment tree from a sequence of points and parent index.
arb::segment_tree segments_from_points(std::vector<arb::mpoint> points,
                                       std::vector<arb::msize_t> parents,
                                       std::vector<int> tags={});

struct cable_cell_description {
    morphology morph;
    label_dict labels;
    decor decorations;

    operator cable_cell() const {
        return cable_cell(morph, decorations, labels);
    }
};

class soma_cell_builder {
    segment_tree tree;
    std::vector<msize_t> branch_distal_id;
    std::unordered_map<std::string, int> tag_map;
    mlocation_list cv_boundaries;
    int tag_count = 0;
    int n_soma_children = 0;

    // Get tag id of region.
    // Add a new tag if region with that name has not already had a tag associated with it.
    int get_tag(const std::string& name);

public:
    soma_cell_builder(double r);

    // Add a new branch that is attached to parent_branch.
    // Returns the id of the new branch.
    msize_t add_branch(msize_t parent_branch, double len, double r1, double r2, int ncomp,
                    const std::string& region);

    mlocation location(mlocation) const;
    mcable cable(mcable) const;

    cable_cell_description make_cell() const;
};

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: HH (default params)
 *    bulk resistivitiy: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Stimuli:
 *    soma centre, t=[10 ms, 110 ms), 0.1 nA
 */

cable_cell_description make_cell_soma_only(bool with_stim = true);

/*
 * Create cell with a soma and unbranched dendrite:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 200 µm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

cable_cell_description make_cell_ball_and_stick(bool with_stim = true);

/*
 * Create cell with a soma and three-branch dendrite with single branch point:
 *
 * O----======
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrites:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 100 µm
 *    compartments: 4
 *
 * Stimulus:
 *    end of first terminal branch, t=[5 ms, 85 ms), 0.45 nA
 *    end of second terminal branch, t=[40 ms, 50 ms), -0.2 nA
 */

cable_cell_description make_cell_ball_and_3stick(bool with_stim = true);

} // namespace arb
