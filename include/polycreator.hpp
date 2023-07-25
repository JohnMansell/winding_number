//
// Created by john on 7/20/23.
//

#ifndef WINDING_NUMBER_HOMEWORK_POLYCREATOR_HPP
#define WINDING_NUMBER_HOMEWORK_POLYCREATOR_HPP

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

#endif  // WINDING_NUMBER_HOMEWORK_POLYCREATOR_HPP
