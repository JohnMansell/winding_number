
#ifndef POLY_CREATOR_HPP
#define POLY_CREATOR_HPP

#include "poly_io.hpp"

namespace poly {

    class Poly_Creator {

    public:
        Poly_Creator() = default;
        Polygon construct_poly(const std::vector<float> &x_vec, const std::vector<float> &y_vec);
        Polygon construct_poly(const std::vector<Point> &points);
        void cleanup_polygon(Polygon &poly);

    };

} // --- END namespace poly

#endif  // POLY_CREATOR_HPP
