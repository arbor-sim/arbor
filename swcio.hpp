#pragma once

#include <exception>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace neuron
{

namespace io
{


class cell_record 
{
public:

    // FIXME: enum's are not completely type-safe, since they can accept anything
    // that can be casted to their underlying type.
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
    
    cell_record()
        : type_(cell_record::undefined)
        , id_(0)
        , x_(0)
        , y_(0)
        , z_(0)
        , r_(0)
        , parent_id_(0)
    { }

    cell_record(const cell_record &other) = default;
    cell_record &operator=(const cell_record &other) = default;

    kind type()
    {
        return type_;
    }

    int id()
    {
        return id_;
    }

    int parent_id()
    {
        return parent_id_;
    }

    float x()
    {
        return x_;
    }

    float y()
    {
        return y_;
    }

    float z()
    {
        return z_;
    }

    float radius()
    {
        return r_;
    }

    float diameter()
    {
        return 2*r_;
    }

private:
    kind type_;         // cell type
    int id_;            // cell id
    float x_, y_, z_;   // cell coordinates
    float r_;           // cell radius
    int parent_id_;     // cell parent's id
};

class swc_parser
{
public:
    swc_parser(const std::string &delim,
               char comment_prefix,
               std::size_t max_fields,
               std::size_t max_line)
        : delim_(delim)
        , comment_prefix_(comment_prefix)
        , max_fields_(max_fields)
        , max_line_(max_line)
    {
        init_linebuff();
    }
               

    swc_parser()
        : delim_(" ")
        , comment_prefix_('#')
        , max_fields_(7)
        , max_line_(256)
    {
        init_linebuff();
    }

    ~swc_parser()
    {
        delete[] linebuff_;
    }

    cell_record parse_record(std::istream &is)
    {
        while (!is.eof() && !is.bad()) {
            // consume empty and comment lines first
            is.getline(linebuff_, max_line_);
            if (linebuff_[0] && linebuff_[0] != comment_prefix_)
                break;
        }

        if (is.eof())
            throw std::runtime_error("unexpected eof found");

        if (is.bad())
            throw std::runtime_error("i/o error");

        if (is.fail() && is.gcount() == max_line_ - 1)
            throw std::runtime_error("too long line detected");

        std::istringstream line(linebuff_);
        return parse_record(line);
    }

private:
    void init_linebuff()
    {
        linebuff_ = new char[max_line_];
    }

    void check_parse_status(const std::istream &is)
    {
        if (is.fail())
            // If we try to read past the eof; fail bit will also be set
            // FIXME: better throw a custom parse_error exception
            throw std::logic_error("could not parse value");

        if (is.bad())
            throw std::runtime_error("i/o error");
    }

    template<typename T>
    T parse_value_strict(std::istream &is)
    {
        T val;
        is >> val;
        check_parse_status(is);

        // everything's fine
        return val;
    }

    // Read the record from a string stream; will be treated like a single line
    cell_record parse_record(std::istringstream &is);

    std::string delim_;
    char comment_prefix_;
    std::size_t max_fields_;
    std::size_t max_line_;
    char *linebuff_;
};


// specialize parsing for cell types
template<>
cell_record::kind swc_parser::parse_value_strict(std::istream &is)
{
    int val;
    is >> val;
    check_parse_status(is);

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
    cell = parser.parse_record(is);
    return is;
}

std::ostream &operator<<(std::ostream &os, const cell_record &cell);


}   // end of neuron::io
}   // end of neuron
