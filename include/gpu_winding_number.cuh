//
// Created by john on 7/23/23.
//

#ifndef GPU_WINDING_NUMBER_CUH
#define GPU_WINDING_NUMBER_CUH

#include "poly_io.hpp"

namespace poly {

class GPU_Winding_Number_Solver {
public:
    float CalculateWindingNumber2D(float *x_vals, float *y_vals, int *results, const Polygon &polygon, size_t count_points);
};

}

#endif  // GPU_WINDING_NUMBER_CUH
