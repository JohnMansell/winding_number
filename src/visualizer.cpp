//
// Created by john on 7/18/23.
//
#ifndef __VISUALIZER__H_
#define __VISUALIZER__H_

#include <SFML/Graphics.hpp>
#include "Visualizer.h"
#include "logging.h"

using namespace std;


Visualizer::Visualizer()
{
    // --- Init Window
    window.create(sf::VideoMode(800, 600), "Polygons");
    scaleFactor = sf::Vector2f(100, 100);

    // --- Init Origin
    origin = sf::CircleShape(3);
    origin.setPosition({400, 300});
    origin.move(-origin.getRadius(), -origin.getRadius());
    origin.setFillColor(sf::Color::Black);

    // --- Init X-Axis
    x_axis = sf::VertexArray(sf::Lines, 2);
    x_axis[0].position = sf::Vector2f(0.f, window.getSize().y / 2);
    x_axis[1].position = sf::Vector2f(window.getSize().x, window.getSize().y / 2);

    // --- Init Y-Axis
    y_axis = sf::VertexArray(sf::Lines, 2);
    y_axis[0].position = sf::Vector2f(window.getSize().x / 2.f, 0.f);
    y_axis[1].position = sf::Vector2f(window.getSize().x / 2.f, window.getSize().y);

    // --- Set Axis Color
    x_axis[0].color = sf::Color::Green;
    x_axis[1].color = sf::Color::Green;
    y_axis[0].color = sf::Color::Green;
    y_axis[1].color = sf::Color::Green;

    // --- Init Point
    point = sf::CircleShape(5);
    point.setFillColor(sf::Color::Magenta);

}

Visualizer::~Visualizer() {
    if (window.isOpen())
        window.close();
}


void Visualizer::new_test_point(float x, float y, std::optional<int> winding) {

    // --- Get Window Dims
    sf::Vector2f centerPoint = origin.getPosition();
    centerPoint.x += origin.getRadius() - point.getRadius();
    centerPoint.y += origin.getRadius() - point.getRadius();

    // --- Set Position
    centerPoint.x += x * scaleFactor.x;
    centerPoint.y += y * scaleFactor.y * -1;
    point.setPosition(centerPoint);


    // --- Set color based on winding number
    sf::Color fill_color = sf::Color::Yellow;
    if(winding)
    {
        if (*winding > 0)
            fill_color = sf::Color::Green;

        else if (*winding < 0)
            fill_color = sf::Color::Cyan;

        else // winding == 0
            fill_color = sf::Color::Red;
    }

    point.setFillColor(fill_color);
}

void Visualizer::show() {

    // --- Open Window
    while (window.isOpen())
    {
        // --- Process Events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // --- Close
            if (event.type == sf::Event::Closed)
                window.close();

            // --- Mouse : Press
            if (event.type == sf::Event::MouseButtonPressed) {
                return;
            }
        }

        // --- Outside of event loop
        window.clear();
        window.draw(*drawable);
        window.draw(x_axis);
        window.draw(y_axis);
        window.draw(origin);
        window.draw(point);

        window.display();
    }
}

void Visualizer::new_point_polygon(const Polygon &poly) {
    __logInfo("\tPoint_Poly : (%.4f, %.4f)\n", poly.x_vec_[0], poly.y_vec_[0]);

    auto single_point_poly = std::make_unique<sf::CircleShape>(7);

    sf::Vector2f p1 = origin.getPosition();
    p1.x += origin.getRadius() - single_point_poly->getRadius();
    p1.y += origin.getRadius() - single_point_poly->getRadius();

    p1.x += poly.x_vec_[0] * scaleFactor.x;
    p1.y += poly.y_vec_[0] * scaleFactor.y * -1;

    (*single_point_poly).setPosition(p1);
    (*single_point_poly).setFillColor(sf::Color::Blue);

    drawable = std::move(single_point_poly);
}


void Visualizer::new_line_polygon(const Polygon& poly) {

    __logInfo("\tLine_Poly : (%0.2f, %0.2f) -> (%0.2f, %0.2f)\n",
           poly.x_vec_[0], poly.y_vec_[0], poly.x_vec_[1], poly.y_vec_[1]);

    auto line = std::make_unique<sf::VertexArray>(sf::Lines, 2);

    // --- Create end points for line
    sf::Vector2f p1 = origin.getPosition();
    sf::Vector2f p2 = origin.getPosition();

    p1.x += origin.getRadius();
    p2.x += origin.getRadius();

    p1.y += origin.getRadius();
    p2.y += origin.getRadius();

    p1.x += poly.x_vec_[0] * scaleFactor.x;
    p1.y += poly.y_vec_[0] * scaleFactor.y * -1;

    p2.x += poly.x_vec_[1] * scaleFactor.x;
    p2.y += poly.y_vec_[1] * scaleFactor.y * -1;

    (*line)[0].position = p1;
    (*line)[1].position = p2;

    (*line)[0].color = sf::Color::Blue;
    (*line)[1].color = sf::Color::Blue;

    drawable = std::move(line);
}


void Visualizer::new_polygon(const Polygon& poly) {

    // --- Set Title
    if(not poly.title.empty())
        window.setTitle(poly.title.data());

    // --- Special Case : Polygon is single point
    if (poly.size() == 1) {
        new_point_polygon(poly);
        return;
    }

    // --- Special Case : Polygon is a Line
    if (poly.size() == 2) {
        new_line_polygon(poly);
        return;
    }

    // --- Normal Case : Polygon
    auto shape = sf::ConvexShape();

    // --- Print Points to Terminal
    #ifdef __DEBUG__
    for(size_t i=0; i<poly.size(); i++)
        cout << "(" << poly.x_vec_[i] << "," << poly.y_vec_[i] << ") ";
    cout << endl;
    #endif

    // --- Shapes
    shape.setPointCount(poly.size());
    shape.setFillColor(sf::Color::Blue);

    auto x_origin = origin.getPosition().x + origin.getRadius();
    auto y_origin = origin.getPosition().y + origin.getRadius();

    sf::Vector2f new_point;
    for (size_t i=0; i<poly.size(); i++)
    {
        new_point.x = x_origin + (scaleFactor.x * poly.x_vec_[i]);
        new_point.y = y_origin + (scaleFactor.y * poly.y_vec_[i] * -1);

        shape.setPoint(i, new_point);
    }

    drawable = std::make_unique<sf::ConvexShape>(shape);
}


#endif // __VISUALIZER__H_