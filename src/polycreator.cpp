//
// Created by john on 7/20/23.
//

#include <deque>
#include <c++/12/iostream>
#include "polycreator.hpp"


namespace poly {

/**
 * Construct a Polygon from two vectors of x and y coordinates. The size of both vectors
 * must be the same, otherwise, a runtime_error is thrown.
 *
 * @param x_vec A vector of x-coordinates for each point.
 * @param y_vec A vector of y-coordinates for each point.
 * @return A Polygon instance containing the points defined by the x and y coordinates.
 * @throws std::runtime_error if x_vec and y_vec are not the same size.
 */
Polygon Poly_Creator::construct_poly(const std::vector<float> &x_vec, const std::vector<float> &y_vec)
{
    // --- Error Checking
    //     : sizes must be equal
    if(x_vec.size() != y_vec.size())
        throw std::runtime_error("Unequal number of X,Y values.");

    std::vector<Point> points;
    points.reserve(x_vec.size());
    for (size_t i=0; i<x_vec.size(); i++)
        points.emplace_back(x_vec[i], y_vec[i]);

    return construct_poly(points);
}

/**
 * Constructs a polygon from a single point.
 *
 * @param points A vector containing a single point.
 * @return A Polygon instance containing the single point.
 */
Polygon construct_single_point_polygon(const std::vector<Point> &points) {
    Polygon point_poly;
    point_poly.x_vec_.push_back(points[0].x);
    point_poly.y_vec_.push_back(points[0].y);

    return point_poly;
}

/**
 * Constructs a polygon that represents a line from two points.
 *
 * @param points A vector containing two points.
 * @return A Polygon instance representing a line between the two points.
 */
Polygon construct_Line(const std::vector<Point> &points) {
    Polygon line_poly;
    line_poly.x_vec_.push_back(points[0].x);
    line_poly.y_vec_.push_back(points[0].y);

    line_poly.x_vec_.push_back(points[1].x);
    line_poly.y_vec_.push_back(points[1].y);

    return line_poly;
}


/**
 * Constructs a Polygon from a vector of points.
 * Cleans the input points to remove redundant points (3+ points in a line), and points that
 * traverse portions of the polygon backwards.
 *
 * @param points A vector of points.
 * @return A Polygon instance containing clean non-redundant points.
 */
Polygon Poly_Creator::construct_poly(const std::vector<Point> &points) {

    // --- Special Case : Single Point
    if (points.size() == 1)
        return construct_single_point_polygon(points);

    // --- Special Case : Line
    if (points.size() == 2)
        return construct_Line(points);

    // --- Sanitize points while building polygon
    std::unordered_map<Point, std::shared_ptr<Vertex>, Point::Hash> visited;

    // --- Initial State
    std::shared_ptr<Vertex> previous;
    std::vector<Point> clean_points;

    // --- First Pass : Create neighbor graph from Vertex(points)
    for (auto &point: points)
    {
        // --- Skip Duplicates < Resolution
        if (previous && (point == previous->point()))
            continue;

        // --- Create new vertex
        if (visited.count(point) == 0)
            visited[point] = std::make_shared<Vertex>(point);

        std::shared_ptr<Vertex> current = visited[point];

        // --- Always include first point
        if (!previous)
            clean_points.push_back(point);

        // --- Connect previous neighbor
        if (previous and previous->neighbors.count(point) == 0){
            previous->neighbors[point] = current;
            current->neighbors[previous->point()] = previous;

            clean_points.push_back(point);
        }

        // --- Increment Pointer
        previous = current;
    }

    // --- Clean Up : Remove Collinear Points
    auto it = clean_points.begin();
    while(it != clean_points.end())
    {
        auto point = *it;

        // --- Check if point is in polygon
        if (visited.count(point) == 0) {
            it++;
            continue;
        }

        auto current = visited[point];

        // --- Check if Vertex(point) has neighbors
        if(current->neighbors.size() < 2){
            it++;
            continue;
        }

        auto prev = current->neighbors.begin()->second;
        auto next = (++current->neighbors.begin())->second;

        // --- Remove redundant middle point along line
        if (current->is_colinear(prev, next) && current->neighbors.size() == 2) {

            // --- Connect neighbors
            prev->neighbors[{next->x, next->y}] = next;
            next->neighbors[{prev->x, prev->y}] = prev;

            // --- Remove middle vertex
            prev->neighbors.erase(point);
            next->neighbors.erase(point);
            visited.erase(point);

            it = clean_points.erase(it);
        }
        else {
            it++;
        }
    }

    // --- Create Polygon from "clean" points
    Polygon poly(visited.size());
    std::shared_ptr<Vertex> prev;
    for (const auto &point: clean_points) {

        // --- Skip if this point was removed in the cleanup process
        if (visited.count(point) == 0) {
            continue;
        }

        auto current = visited[point];

        // --- Skip immediate duplicates
        if (prev and (prev->point() == current->point()))
            continue;

        poly.x_vec_.push_back(point.x);
        poly.y_vec_.push_back(point.y);

        prev = current;
    }

    // --- Don't 'close' lines
    if(poly.size() == 2) {
        return poly;
    }

    // --- Close Polygon
    poly.ClosePolygon();
    return poly;
}


} // --- END namespace poly


#include "polycreator.hpp"
