#pragma once

// Maintain a pool of morphologies for use with miniapp recipes.
// The default pool comprises a single ball-and-stick morphology;
// sets of morphologies can be loaded from SWC files.

#include <memory>
#include <string>
#include <vector>

#include <morphology.hpp>
#include <util/path.hpp>

namespace arb {

class morphology_pool {
    std::shared_ptr<std::vector<morphology>> pool;

public:
    // Construct default empty pool.
    morphology_pool(): pool(new std::vector<morphology>) {}

    // Construct pool with one starting morphology.
    explicit morphology_pool(morphology m): pool(new std::vector<morphology>) {
        insert(std::move(m));
    }

    std::size_t size() const { return pool->size(); }
    const morphology& operator[](std::ptrdiff_t i) const { return (*pool)[i]; }

    void insert(morphology m) { (*pool).push_back(std::move(m)); }
    void clear() { (*pool).clear(); }
};

extern morphology_pool default_morphology_pool;

void load_swc_morphology(morphology_pool& pool, const util::path& swc_path);
void load_swc_morphology_glob(morphology_pool& pool, const std::string& pattern);

} // namespace arb
