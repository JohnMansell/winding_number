# Dyndrite Take Home Assignment

# Overview

The goal of this assignment is to implement a “winding number” algorithm which determines the number of times a polygon winds around a given test point. The test point, and the vertices of the polygon are given in (x,y) coordinates.

### Objectives

- Implement the winding number algorithm by subclassing the `IWindingNummberAlgorithm` interface class.
- Improve the `DefaultPolygonReader`class to be more resilient to bad input.

# Requirements

The solution has been implemented on the following system. It will be relatively portable to other systems, but has not yet been tested and validated for portability. 

- Fedora 38
- g++ 13.1.1
- C++ 23
- nvcc 12.1
- CUDA 12.1

An additional library (SFML) was added to enable visualization of the polygons and points. The SFML library can be downloaded and installed [here](https://www.sfml-dev.org/download.php). Or using :

```bash
# Fedora
> sudo dnf install SFML-devel

# Ubuntu & Debian
> sudo apt-get install libsfml-dev
```

[2.6 Tutorials (SFML / Learn)](https://www.sfml-dev.org/tutorials/2.6/)

# Winding Number

## Intro

[Winding number](https://en.wikipedia.org/wiki/Winding_number)

[Inclusion of a Point in a Polygon](https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html)

The primary goal of the library is to determine the winding number of a point relative to a polygon. The winding number is the number of times a polygon wraps counterclockwise around the point. This is a useful way to determine if a point is located inside the polygon or not. This is effective because the winding number method allows for non-simple polygons with holes, as well as complex curves where it would otherwise be computationally costly to determine the location of the point to the polygon or curve.

For simple polygons, without holes or with a smaller number of vertices, there are simpler means to solve the “point-in-polygon” problem. But the winding number solution is computationally equivalent for simple polygons, and is much more robust.

## Implementation

My C++ algorithm was based on the method described by [Dan Sunday](https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html). I also expanded the functionality of the class to check for the type of polygon being tested, to account for other cases. Specifically, my algorithm first tests if the polygon has only 1, 2, or 3 vertices.

1. A polygon with only one vertex is a single point. Therefor rather than checking for the winding number the function tests whether the test point has the same coordinates as the polygon vertex, within a certain tolerance.

2. A polygon with 2 vertices is a line. We can therefor use simple equations to test if the test point falls along that line, and is on the line segment or not. Again, we allow for a certain tolerance of deviation.

3. A polygon with 3 vertices is a triangle. This also allows us to use simpler formulas to determine if the test point is contained within the area of the triangle. It’s also worth noting that this would be especially valuable if the class were extended to account for 3-dimensional figures, where it would likely be deconstructed into a set of 2D triangles covering the surface of the figure.

If the polygon has more than 3 vertices, the winding number is determined using the algorithm described by Dan Sunday which looks at the crossing direction of each line segment for each pair of vertices in order.

```cpp
int CalculateWindingNumber2D(float x, float y, Polygon polygon)
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
```

## Challenges

Although I am able to stand on the shoulders of giants who came before me who have done incalculable work to refine the techniques for solving these problems; there are still challenges to solve when seeking to optimize the winding number algorithm.

**Bounding Box Optimization**

For example, it must be determined on a case-by-case basis if a bounding box should be found before solving for the winding number. This is a trade off because finding a bounding box can be as computationally expensive as solving for the winding number. In that case, if only a few points are to be tested, it is not generally worthwhile to find the bounding box ahead of time. Conversely, if many points will be used, or if the polygon will be reused multiple times throughout the life of the program, it is vastly beneficial to first find the smallest bounding box (or other shape) that contains the polygon in question. From there, multiple points can be rapidly eliminated from inclusion, because they will fall outside the boundary, and therefor cannot be a part of the polygon.

I opted to not solve the bounding box problem for a few reasons. First, the polygons are being tested against a single point, rather than many. Second, the bounding box solution would need likely be implemented differently for various real world applications, and thus would not apply directly. Various types of 3D figures for example yield better or worse optimizations when projecting onto a 2D plane. The kind of bounding box solution applied would need to be case specific, and I would rather build out the solver class to handle 3D cases first, and then implement the appropriate bounding box solution.

## Further Optimizations

It would be very interesting to flush out the library to solve for more complex real-world problems. I think the next exercise I would attempt would be to solve for the winding number inside a complex curve to test for inclusion. This could either be simplified to resemble a polygon, or a selection of sample points could be used to approximate the shape of the curve. However, these would likely involve trig functions, and I would need to test the actual performance of each case to see which performs better.

Next, it would be great to extend the solver to a 3D class which can find whether or not the test point is inside the volume or surface area of a figure. From the reading I’ve done, finding an optimized approach to projecting the point onto a 2D plane using transformations would be a challenging and rewarding task. Specifically, I would like to see how optimized the performance can get on a CUDA capable GPU; especially if the tensor cores are utilized since the transformation matrix dimensions would be small enough to fit inside, and the hardware optimizations would allow for very high throughput. I’d also find it particularly interesting because my main niche has become optimizing massively parallel computations of large datasets.

# Polygon Reader

Typically, if there is file IO to be done, I prefer to do so in python. The duck typing of python combined with the built in functionality leads (IMHO) to much more resilient code. Since file IO is typically only done once at the beginning of a program, the trade-off in speed is worthwhile. Once the objects are in memory, they can easily be accessed by the much faster C or C++ libraries for actual computation. It also tends to make a much more user friendly wrapper around the whole class enabling the end-user to feed custom data and test functionality without needing to dive into statically typed languages; but they still get the performance of highly optimized code.

That being said, I’ve added a number of checks I believe contribute to the resiliency of the Polygon Reader class, and also improve the performance of the Winding Number algorithm.

1. **Removing duplicate points.**
If the user has entered the same point twice consecutively, there is no benefit to including it twice since the winding number will not change at all traversing from point A —> B. This also prevents the need to check for distance between the two points being zero, which can cause “division by zero” errors in the winding number algorithm when solving for area, for example.

2. **Removing redundant collinear points**
I also removed redundant points; by which I mean any points along a line segment which are not connected to any points not on the same line segment. Defining multiple points on a line segment only adds to the number of iterations for the winding number algorithm, and will not change the result. Only the two end points are useful information.

3. **Removing “backwards” traversal points**
In one of the examples, the vertices of a triangle are given, and then the same vertices are supplied but in reverse order. The polygon effectively travels once around the perimeter, and then a second time around the perimeter in reverse order. By storing the neighbors of each vertex in a map, I was able to avoid adding duplicate connections between the same two vertices. 

This not only means less computations for the winding number algorithm, but also gives the intuitive answer for inclusion. Winding once around a point, and then backwards around the point would give a winding number of zero by definition. This kind of edge behavior should always be evaluated to ensure the function is producing the expected result. It may be the case that there are valid reasons for defining a polygon this way. For example, this could be the method of defining a completely hollow polygon. I made the decision to treat this kind of input as corrupted data, and simplified the polygon to be a triangle with only one trip around the perimeter. There are much simpler and computationally efficient ways to define an empty polygon, which could be an extension of the polygon class and would not require the winding number algorithm to compute 2x the number of vertices.

4. ************************************************Even number of X:Y pairs************************************************
There are infinitely many ways a user could supply unexpected, or “bad” input data. One way we can validate the data ahead of time is to ensure that for every coordinate provided, the user has supplied both a valid ‘x’ and ‘y’ coordinate. If there the input length is uneven, the final point will be undefined. There are multiple ways to handle this, but I find it best to throw the error early and give the user the opportunity to correct the data, rather than supplying a winding number that may not be the solution the user was expecting.

# CUDA GPU Winding Number

Though not officially part of the assignment, I was curious to see the performance of a basic CUDA kernel implementing the same logic for the winding number algorithm. 

|  | Time (ms) | N-Points | N-Polygons |
| --- | --- | --- | --- |
| CPU Algorithm | 1414 ms | 2 e 16 | 36 |
| GPU Kernel | 0.576 ms | 2 e 16 | 36 |

Even on a modest GTX 1060, the GPU is able to out perform the linear algorithm by multiple orders of magnitude. Obviously this would need to be optimized for a specific case load, but testing 1<<16 (x,y) pairs (65536 points) against 36 polygons, the CUDA kernel is able to complete the point-in-polygon problem in 576 microseconds, vs 1400 milliseconds on the CPU (Intel 6700). 

## Implementation

Each block in the grid loads the vertices of the polygon into shared memory, and then each thread calculates the winding number for one point relative to the polygon.

We are able to use the same methods for testing single-point, line, and triangle polygons as in the linear version without worrying about warp-divergence because the block is all evaluating the same polygon, so the same branch will be followed for every thread.

A simple optimization to add would be to copy all the polygons to the device initially. Each block could then either evaluate all points vs a single polygon; or each block could evaluate all polygons for some subset of points using a block stride grid to optimize memory access.
The design would depend on the size and number of the polygons, and the number of points to be tested.