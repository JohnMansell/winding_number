#include "poly_io.hpp"
#include "winding.hpp"
#include "Visualizer.h"
#include "common.cuh"
#include "logging.h"
#include "gpu_winding_number.cuh"

#include <iostream>
#include <bits/stdc++.h>

using namespace poly;
using namespace std;
using namespace std::chrono;

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

typedef std::tuple<float, float, Polygon> point_n_poly;

double cpu_batch_timing(float *x_vals, float *y_vals, std::vector<point_n_poly> polygons, int *results, size_t count)
{
    // --- Initialize
    auto cpu = winding_number::IWindingNumberAlgorithm::Create();
    auto start = high_resolution_clock::now();

    // --- Winding number for all points for all polygons
    for(const auto &[x, y, poly]: polygons)
    {
        for(size_t i=0; i<count; i++)
            results[i] = *cpu->CalculateWindingNumber2D(x_vals[i], y_vals[i], poly);
    }

    // --- Microseconds
    auto stop = high_resolution_clock::now();
    return duration<double, milli>(stop - start).count();
}


int main(int nargs, char* args[], char* env[]) {

    // --- Read Polygons from File
    auto poly_reader = IPolygonReader::Create();
    const auto polygons = poly_reader->ReadPointsAndPolygonsFromFile("../test/polygons.txt");
    auto winding_algo = winding_number::IWindingNumberAlgorithm::Create();
    auto gpu_solver = GPU_Winding_Number_Solver();

    // --- Display Polygons
    auto visualizer = Visualizer();
    std::optional<int> wind;
    for(const auto &[x, y, poly] : polygons)
    {
        wind = winding_algo->CalculateWindingNumber2D(x, y, poly);
        if (wind)
            __logDebug("%s Winding number = %d", poly.title.c_str(), *wind);
        else
            __logError("Error = %s\n", winding_algo->error_message().c_str());

        visualizer.new_polygon(poly);
        visualizer.new_test_point(x, y, wind);
        visualizer.show();
    }

    // --- Initialize Random Generator
    default_random_engine gen;
    uniform_real_distribution<> distribution(0.5, 1.0);

    // --- Allocate Managed Memory
    const size_t N = 1 << 16;
    float *x_vals = __gpuMallocManaged(N);
    float *y_vals = __gpuMallocManaged(N);
    int cpu_results[N] = {0};
    int *gpu_results = __gpuMallocManaged(N);

    // --- Generate Random Values
    for(int i=0; i<N; i++)
    {
        x_vals[i] = distribution(gen);
        y_vals[i] = distribution(gen);
    }

    // --- CPU Batch Timing
    double cpu_total = cpu_batch_timing(x_vals, y_vals, polygons, cpu_results, N);

    // --- Prefetch Data to GPU
    int device = 0;
    cudaMemPrefetchAsync(x_vals, sizeof(float) * N, device, nullptr);
    cudaMemPrefetchAsync(y_vals, sizeof(float) * N, device, nullptr);
    __gpuErrCheck();

    // --- GPU Batch Timing
    float gpu_total = 0;
    for (const auto &[x, y, poly] : polygons)
    {
        gpu_total += gpu_solver.CalculateWindingNumber2D(x_vals, y_vals, gpu_results, poly, N);
    }

    __logAlways("CPU Time(%cs) = %11.2f/(%lu points * %d polygons). Ave = %f", 0xCE, cpu_total, N, polygons.size(), cpu_total / (N * polygons.size()));
    __logAlways("GPU Time(%cs) = %11.2f/(%lu points * %d polygons). Ave = %f", 0xCE, gpu_total, N, polygons.size(), gpu_total / (N * polygons.size()));

    return 0;
}