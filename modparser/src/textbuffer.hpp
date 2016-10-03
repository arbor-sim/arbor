#pragma once

#include <limits>
#include <sstream>
#include <string>

class TextBuffer {
public:
    TextBuffer() {
        text_.precision(std::numeric_limits<double>::max_digits10);
    }
    TextBuffer& add_gutter();
    void add_line(std::string const& line);
    void add_line();
    void end_line(std::string const& line);
    void end_line();

    std::string str() const;

    void set_gutter(int width);

    void increase_indentation();
    void decrease_indentation();
    std::stringstream &text();

private:

    int indent_ = 0;
    const int indentation_width_=4;
    std::string gutter_ = "";
    std::stringstream text_;
};

template <typename T>
TextBuffer& operator<< (TextBuffer& buffer, T const& v) {
    buffer.text() << v;

    return buffer;
}

