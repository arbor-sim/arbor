* change `probe_sort` enum to scoped enum
    - renamed to `probeKind`
    - refactored out of class
    - updated coding guidelines wiki with enum rules
* refactor `std::pair` into structs with meaningfull name types in `cell.hpp`
    - not just probes: stimulii and detectors too.
* add profiling region around sampling in `cell_group`
* change output data format for traces to json
* remove white space at end of lines (looking at you Sam)
