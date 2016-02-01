#pragma once

#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace nestmc
{

namespace io
{


static bool starts_with(const std::string &str, const std::string &prefix)
{
    return (str.find(prefix) == 0);
}

class cell_record 
{
public:
    using id_type = int;

    // FIXME: enum's are not completely type-safe, since they can accept
    // anything that can be casted to their underlying type.
    // 
    // More on SWC files: http://research.mssm.edu/cnic/swc.html
    enum kind {
        undefined = 0,
        soma,
        axon,
        dendrite,
        apical_dendrite,
        fork_point,
        end_point,
        custom
    };

    // cell records assume zero-based indexing; root's parent remains -1
    cell_record(kind type, int id, 
                float x, float y, float z, float r,
                int parent_id)
        : type_(type)
        , id_(id)
        , x_(x)
        , y_(y)
        , z_(z)
        , r_(r)
        , parent_id_(parent_id)
    {
        check_consistency();
    }
    
    cell_record()
        : type_(cell_record::undefined)
        , id_(0)
        , x_(0)
        , y_(0)
        , z_(0)
        , r_(0)
        , parent_id_(-1)
    { }

    cell_record(const cell_record &other) = default;
    cell_record &operator=(const cell_record &other) = default;

    // Equality and comparison operators
    friend bool operator==(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return lhs.id_ == rhs.id_;
    }

    friend bool operator<(const cell_record &lhs,
                          const cell_record &rhs)
    {
        return lhs.id_ < rhs.id_;
    }

    friend bool operator<=(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return (lhs < rhs) || (lhs == rhs);
    }

    friend bool operator!=(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator>(const cell_record &lhs,
                          const cell_record &rhs)
    {
        return !(lhs < rhs) && (lhs != rhs);
    }

    friend bool operator>=(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return !(lhs < rhs);
    }

    friend std::ostream &operator<<(std::ostream &os, const cell_record &cell);

    kind type() const
    {
        return type_;
    }

    id_type id() const
    {
        return id_;
    }

    id_type parent() const
    {
        return parent_id_;
    }

    float x() const
    {
        return x_;
    }

    float y() const
    {
        return y_;
    }

    float z() const
    {
        return z_;
    }

    float radius() const
    {
        return r_;
    }

    float diameter() const
    {
        return 2*r_;
    }

    void renumber(id_type new_id, std::map<id_type, id_type> &idmap)
    {
        auto old_id = id_;
        id_ = new_id;

        // Obtain parent_id from the map
        auto new_parent_id = idmap.find(parent_id_);
        if (new_parent_id != idmap.end()) {
            parent_id_ = new_parent_id->second;
        }

        check_consistency();
        idmap.insert(std::make_pair(old_id, new_id));
    }

private:
    void check_consistency() const
    {
        // Check cell type as well; enum's do not offer complete type safety,
        // since you can cast anything that fits to its underlying type
        if (type_ < 0 || type_ > custom)
            throw std::invalid_argument("unknown cell type");

        if (id_ < 0)
            throw std::invalid_argument("negative ids not allowed");
        
        if (parent_id_ < -1)
            throw std::invalid_argument("parent_id < -1 not allowed");

        if (parent_id_ >= id_)
            throw std::invalid_argument("parent_id >= id is not allowed");

        if (r_ < 0)
            throw std::invalid_argument("negative radii are not allowed");
    }

    kind type_;         // cell type
    id_type id_;        // cell id
    float x_, y_, z_;   // cell coordinates
    float r_;           // cell radius
    id_type parent_id_; // cell parent's id
};

class swc_parse_error : public std::runtime_error
{
public:
    explicit swc_parse_error(const char *msg)
        : std::runtime_error(msg)
    { }

    explicit swc_parse_error(const std::string &msg)
        : std::runtime_error(msg)
    { }
};

class swc_parser
{
public:
    swc_parser(const std::string &delim,
               std::string comment_prefix)
        : delim_(delim)
        , comment_prefix_(comment_prefix)
    { }

    swc_parser()
        : delim_(" ")
        , comment_prefix_("#")
    { }

    std::istream &parse_record(std::istream &is, cell_record &cell)
    {
        while (!is.eof() && !is.bad()) {
            // consume empty and comment lines first
            std::getline(is, linebuff_);
            if (!linebuff_.empty() && !starts_with(linebuff_, comment_prefix_))
                break;
        }

        if (is.bad())
            // let the caller check for such events
            return is;

        if (is.eof() &&
            (linebuff_.empty() || starts_with(linebuff_, comment_prefix_)))
            // last line is either empty or a comment; don't parse anything
            return is;

        if (is.fail())
            throw swc_parse_error("too long line detected");

        std::istringstream line(linebuff_);
        cell = parse_record(line);
        return is;
    }

private:
    void check_parse_status(const std::istream &is)
    {
        if (is.fail())
            // If we try to read past the eof; fail bit will also be set
            throw swc_parse_error("could not parse value");
    }

    template<typename T>
    T parse_value_strict(std::istream &is)
    {
        T val;
        check_parse_status(is >> val);

        // everything's fine
        return val;
    }

    // Read the record from a string stream; will be treated like a single line
    cell_record parse_record(std::istringstream &is);

    std::string delim_;
    std::string comment_prefix_;
    std::string linebuff_;
};


// specialize parsing for cell types
template<>
cell_record::kind swc_parser::parse_value_strict(std::istream &is)
{
    cell_record::id_type val;
    check_parse_status(is >> val);

    // Let cell_record's constructor check for the type validity
    return static_cast<cell_record::kind>(val);
}

cell_record swc_parser::parse_record(std::istringstream &is)
{
    auto id = parse_value_strict<int>(is);
    auto type = parse_value_strict<cell_record::kind>(is);
    auto x = parse_value_strict<float>(is);
    auto y = parse_value_strict<float>(is);
    auto z = parse_value_strict<float>(is);
    auto r = parse_value_strict<float>(is);
    auto parent_id = parse_value_strict<int>(is);

    // Convert to zero-based, leaving parent_id as-is if -1
    if (parent_id != -1)
        parent_id--;

    return cell_record(type, id-1, x, y, z, r, parent_id);
}


std::istream &operator>>(std::istream &is, cell_record &cell)
{
    swc_parser parser;
    parser.parse_record(is, cell);
    return is;
}


std::ostream &operator<<(std::ostream &os, const cell_record &cell)
{
    // output in one-based indexing
    os << cell.id_+1 << " "
       << cell.type_ << " "
       << std::setprecision(7) << cell.x_ << " "
       << std::setprecision(7) << cell.y_ << " "
       << std::setprecision(7) << cell.z_ << " "
       << std::setprecision(7) << cell.r_ << " "
       << ((cell.parent_id_ == -1) ? cell.parent_id_ : cell.parent_id_+1);

    return os;
}

//
// Reads cells from an input stream until an eof is encountered and returns a
// cleaned sequence of cell records.
//
// For more information check here:
//   https://github.com/eth-cscs/cell_algorithms/wiki/SWC-file-parsing
//
std::vector<cell_record> swc_read_cells(std::istream &is)
{
    std::vector<cell_record> cells;
    std::unordered_set<cell_record::id_type> ids;

    std::size_t          num_trees = 0;
    cell_record::id_type last_id   = -1;
    bool                 needsort  = false;

    cell_record curr_cell;
    while ( !(is >> curr_cell).eof()) {
        if (curr_cell.parent() == -1 && ++num_trees > 1)
            // only a single tree is allowed
            break;

        auto inserted = ids.insert(curr_cell.id());
        if (inserted.second) {
            // not a duplicate; insert cell
            cells.push_back(curr_cell);
            if (!needsort && curr_cell.id() < last_id)
                needsort = true;

            last_id = curr_cell.id();
        }
    }

    if (needsort)
        std::sort(cells.begin(), cells.end());

    // Renumber cells if necessary
    std::map<cell_record::id_type, cell_record::id_type> idmap;
    cell_record::id_type next_id = 0;
    for (auto &c : cells) {
        if (c.id() != next_id)
            c.renumber(next_id, idmap);

        ++next_id;
    }

    return std::move(cells);
}

}   // end of nestmc::io
}   // end of nestmc
