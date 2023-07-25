//
// Created by john on 7/18/23.
//

#ifndef WINDING_NUMBER_HOMEWORK_VISUALIZER_H
#define WINDING_NUMBER_HOMEWORK_VISUALIZER_H

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>
#include <iostream>

#include "poly_io.hpp"


using namespace poly;

class Visualizer {
public:
    Visualizer();
    virtual ~Visualizer();

    void new_polygon(const Polygon & poly);
    void new_line_polygon(const Polygon &poly);
    void new_point_polygon(const Polygon &poly);
    void new_test_point(float x, float y, std::optional<int> winding);
    void show();


private:
    // --- Permanent
    sf::VertexArray x_axis;
    sf::VertexArray y_axis;
    sf::CircleShape origin;
    sf::RenderWindow window;
    sf::Vector2f scaleFactor;

    // --- Variable
    sf::CircleShape point;
    std::unique_ptr<sf::Drawable> drawable;
};

#endif  // WINDING_NUMBER_HOMEWORK_VISUALIZER_H
