#include <iostream>
#include <fstream>
#include <numeric>

#include "cell_decomposer.hpp"
#include "cell_tree.hpp"

#include "json/src/json.hpp"

using json = nlohmann::json;

void test_simple() {
    {
        std::vector<int> parent_index =
            {0, 0, 1, 2, 0, 4};
        cell_tree tree(parent_index);
    }
    {
        std::vector<int> parent_index =
            {0, 0, 1, 2, 0, 4, 0, 6, 7, 8};
        cell_tree tree(parent_index);
    }
    {
        std::vector<int> parent_index =
            {0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12};
        cell_tree tree(parent_index);
        tree.to_graphviz("test.dot");
    }
}

void test_json() {
    using range = memory::Range;
    json  cell_data;
    std::ifstream("cells_small.json") >> cell_data;

    for(auto c : range(0,20)) {
    //for(auto c : range(0,cell_data.size())) {
        std::vector<int> parent_index = cell_data[c]["parent_index"];
        cell_tree tree(parent_index);
        std::cout << "cell " << c << " ";
        tree.balance();
        tree.to_graphviz("cell_" + std::to_string(c) + ".dot");
        std::cout << memory::util::yellow("---------") << std::endl;
    }
}

int main(void)
{
    test_simple();
    test_json();

    return 0;
}
