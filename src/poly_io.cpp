#include <poly_io.hpp>

#include <format>
#include <cassert>
#include <cmath>
#include <filesystem>  // A C++17 capable compiler is assumed here.
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <iostream>
#include <cstdio>
#include "poly_creator.hpp"
#include "logging.h"

// This makes it easy and consistent to include std::filesystem, which
// for older compilers is not always in <filesystem> or in std::filesystem
#if defined(_WIN32)
#    if _MSC_VER >= 1914
#        include <filesystem>
namespace fs = std::filesystem;
#    else
#        include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#    endif
#elif __has_include(<filesystem>)
#    include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#elif defined(__GNUC__) && __GNUC__ <= 7
#    include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#elif defined(__GNUC__) && __GNUC__ >= 8
#    include <filesystem>
namespace fs = std::filesystem;
#elif defined(__clang__) && __clang_major__ <= 8
#    include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#elif defined(__clang__) && __clang_major__ <= 10
#    include <filesystem>
namespace fs = std::filesystem;
#endif

namespace poly {

namespace {

    using namespace std;

    ostream& operator<<(ostream& os, const Polygon& poly) {
        for (size_t i=0; i<poly.size(); i++)
            os << "(" << poly.x_vec_[i] << "," << poly.y_vec_[i] << ") ";

        return os;
    }

    template <class... Args>
    ostream& operator<<(ostream& os, tuple<Args...> const& t) {
        os << "(";
        bool first = true;
        apply([&os, &first](auto&&... args) {
            auto print = [&] (auto&& val) {
                if (!first)
                    os << ",";
                (os << " " << val);
                first = false;
            };
            (print(args), ...);
        }, t);
        os << " )";
        return os;
    }

    typedef std::tuple<float, float, Polygon> Point_n_Poly;


    // Split a string in to separate strings, separated by any of the passed in delimeters (each is a character).
    std::vector<std::string_view> SplitString(std::string_view to_split, std::string_view delims)
    {
        std::vector<std::string_view> split_strings;

        for (size_t first = 0, second = 0; first < to_split.size() && second != std::string_view::npos;
             first = second + 1)
        {
            second = to_split.find_first_of(delims, first);
            if (first != second) {
                split_strings.emplace_back(to_split.substr(first, second - first));
            }
        }

        return split_strings;
    }

class DefaultPolygonReader : public IPolygonReader {

    private:
        Poly_Creator creator;

    public:
        static float read_float(const std::string_view element_view) {
            std::string element_string;
            element_string = element_view;

            try {
                return std::stof(element_string);
            }

            catch (const std::invalid_argument&) {
                throw std::runtime_error("Could not parse line because this is not a floating point value: " +
                                         element_string);
            }

            catch (const std::out_of_range&) {
                throw std::runtime_error("Could not parse line because this is too large to fit in a float: " +
                                         element_string);
            }
        }

        Point_n_Poly CreatePointAndPolygonFromString(std::string_view polygon_string) override {
            // --- Get elements from string
            auto split_string = SplitString(polygon_string, " ");

            // --- Local Variables
            std::string element_string;
            std::vector<Point> points;
            std::optional<float> x;

            // --- Convert elements to point & polygon
            for (std::string_view element_view : split_string) {
                float value = read_float(element_view);

                // --- Initialize X First
                if (not x) {
                    x = value;
                    continue;
                }

                else {
                    points.emplace_back(*x, value);
                    x.reset();
                }
            }

            // --- Error Checking
            //     : Test Point (x,y) exists
            if (points.empty()) {
                if (not x)
                    throw std::runtime_error("Missing initial x-value for point.");

                else
                    throw std::runtime_error("Missing initial y-value for point.");
            }

            // --- Error Checking
            //     : Unequal Number of X:Y values
            if (x)
                throw std::runtime_error("Unequal number of X:Y values");

            // --- Pop 'test_point'
            Point test_point = points.at(0);
            points.erase(points.begin());

            // --- Create and Return
            return {test_point.x, test_point.y, creator.construct_poly(points)};
        }

        std::vector<Point_n_Poly> ReadPointsAndPolygonsFromFile(std::string_view filepath) override {

            // --- Error Checking
            //     : Validate File Path
            fs::path path(filepath);
            if (!fs::exists(path) || !fs::is_regular_file(path)) {
                throw std::runtime_error("Provided filepath is not readable as a file: " + std::string(filepath));
            }

            // --- Local Variables
            std::vector<Point_n_Poly> point_and_polygons;
            std::string line;
            std::string title;
            std::ifstream fs;

            // --- Read from File
            try
            {
                fs = std::ifstream(path, std::ios::in);
                fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);

                // --- Get Points & Polygons line-by-line
                while (fs.peek() != std::char_traits<char>::eof())
                {
                    std::getline(fs, line);

                    // --- Skip empty lines
                    if(line.empty())
                        continue;

                    try {
                        point_and_polygons.emplace_back(CreatePointAndPolygonFromString(line));
                        if (not title.empty()) {
                            auto& poly = std::get<2>(point_and_polygons.back());
                            poly.title = "" + title;
                        }
                    }

                    catch (const std::runtime_error& err) {
                        __logDebug("Runtime Error : %s", err.what());
                        title = line.c_str();
                        continue;
                    }
                }
            }

            // --- Error Checking
            //     : Failure to read file
            catch (const std::ios_base::failure& e) {
                fs.close();
                throw std::runtime_error("Failed to read:\t" + std::string(filepath) + "\n" +  //
                                         "Error:\t\t" + e.what());
            }

            // --- Error Checking
            //     : Failure to read line
            catch (const std::runtime_error& e) {
                fs.close();
                throw std::runtime_error("Failed to read a line in:\t" + std::string(filepath) + "\n" +  //
                                         "Error:\t\t" + e.what());
            }

            // --- Clean up
            fs.close();
            return point_and_polygons;
        }
};

}  // namespace


Polygon::Polygon(size_t capacity) {
    x_vec_.reserve(capacity);
    y_vec_.reserve(capacity);
}

void Polygon::AppendPoint(float x, float y) {
    x_vec_.push_back(x);
    y_vec_.push_back(y);
}

size_t Polygon::size() const {
    size_t x_vec_size = x_vec_.size();
    assert(x_vec_size == y_vec_.size());
    return x_vec_size;
}

void Polygon::ClosePolygon() {
    if (size() == 0 || IsClosed()) {
        return;
    }
    x_vec_.push_back(x_vec_[0]);
    y_vec_.push_back(y_vec_[0]);
}

bool Polygon::IsClosed(float tolerance) const {
    return (std::abs(x_vec_.front() - x_vec_.back()) <= tolerance &&  //
            std::abs(y_vec_.front() - y_vec_.back()) <= tolerance);
}

std::unique_ptr<IPolygonReader> IPolygonReader::Create() {
    return std::make_unique<DefaultPolygonReader>();
}

}  // namespace poly
