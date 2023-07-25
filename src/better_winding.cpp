#include "winding.hpp"


namespace winding_number {

typedef std::optional<int> opt_int;

using namespace poly;

    class Better_Winding_Number_Algorithm : public IWindingNumberAlgorithm
    {
        static float isLeft(const Point &p0, const Point &p1, const Point &p_test) {
                return (p1.x - p0.x) * (p_test.y - p0.y) - (p_test.x - p0.x) * (p1.y - p0.y);
        }

        static bool isCollinear(const Point &p1, const Point &p2, const Point &p3) {
                // Calculate the area of the triangle formed by p1, p2, p3
                // If the area is 0, then the points are collinear.
                float area = 0.5 * std::abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
                return area == 0;
        }

        static opt_int points_are_equal(const Point &p_test, Polygon &polygon) {
                Point poly_point(polygon.x_vec_[0], polygon.y_vec_[0]);

                // Check if points are within {RESOLUTION} of each other
                return 1 * (p_test == poly_point);
        }

        static opt_int point_on_line(const Point &p_test, const Polygon &polygon) {
                Point p1(polygon.x_vec_[0], polygon.y_vec_[0]);
                Point p2(polygon.x_vec_[1], polygon.y_vec_[1]);

                if (!isCollinear(p_test, p1, p2))
                    return 0;

                // Now check if p_test lies within p1 and p2 on the line
                if (p_test.x <= std::max(p1.x, p2.x) &&
                    p_test.x >= std::min(p1.x, p2.x) &&
                    p_test.y <= std::max(p1.y, p2.y) &&
                    p_test.y >= std::min(p1.y, p2.y))
                    return 1;

                return 0;
        }

        static opt_int point_is_inside_triangle(const Point &p, const Polygon& polygon) {
                Point a(polygon.x_vec_[0], polygon.y_vec_[0]);
                Point b(polygon.x_vec_[1], polygon.y_vec_[1]);
                Point c(polygon.x_vec_[2], polygon.y_vec_[2]);

                float alpha = ((b.y - c.y)*(p.x - c.x) + (c.x - b.x)*(p.y - c.y)) /
                              ((b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y));

                float beta = ((c.y - a.y)*(p.x - c.x) + (a.x - c.x)*(p.y - c.y)) /
                             ((b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y));

                float gamma = 1.0f - alpha - beta;

                // The point is inside the triangle if 0 <= alpha <= 1, 0 <= beta <= 1, 0 <= gamma <= 1
                return (alpha >= 0 && beta >= 0 && gamma >= 0) ? 1 : 0;
        }

        opt_int CalculateWindingNumber2D(float x, float y, Polygon polygon) override
        {
            int wn = 0;
            Point p_test(x, y);

            // --- Special Cases
            switch(polygon.size())
            {
                case 1:
                    return points_are_equal(p_test, polygon);

                case 2:
                    return point_on_line(p_test, polygon);

                case 3:
                    return point_is_inside_triangle(p_test, polygon);
            }


            // --- Loop through all vertices of the polygon
            for (size_t i = 0; i < polygon.size(); i++) {
                Point p0 = {polygon.x_vec_[i], polygon.y_vec_[i]};
                Point p1 = {polygon.x_vec_[i + 1], polygon.y_vec_[i + 1]};

                // --- Upward Crossing
                if (polygon.y_vec_[i] <= y) {
                    if (polygon.y_vec_[i + 1] > y)
                        if (isLeft(p0, p1, p_test) > 0)
                            ++wn;
                }

                // --- Downward Crossing
                else {
                    if (polygon.y_vec_[i + 1] <= p_test.y)
                        if (isLeft(p0, p1, p_test) < 0)
                            --wn;
                }
            }

            return wn;
        }
    };

    std::unique_ptr<IWindingNumberAlgorithm> IWindingNumberAlgorithm::Create() {
        return std::make_unique<Better_Winding_Number_Algorithm>();
    }

    void IWindingNumberAlgorithm::tolerance(float tolerance) noexcept {
        tolerance_ = tolerance;
    }

    float IWindingNumberAlgorithm::tolerance() const noexcept {
        return tolerance_;
    }

    std::string IWindingNumberAlgorithm::error_message() const noexcept {
        return error_message_;
    }

    void IWindingNumberAlgorithm::error_message(std::string error_message) noexcept {
        error_message_ = std::move(error_message);
    }

} // --- END namespace winding number
// --- END namespace poly
