
// --- CUDA
#include "cuda.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

// --- Project Headers
#include "gpu_winding_number.cuh"
#include "common.cuh"
#include "logging.h"

using namespace cooperative_groups;

__device__ __inline__
float is_left(const float2 p0, const float2 p1, const float2 p_test) {
    return (p1.x - p0.x) * (p_test.y - p0.y) - (p_test.x - p0.x) * (p1.y - p0.y);
}

__device__ __inline__
bool isCollinear(const float2 p1, const float2 p2, const float2 p3) {
    // Calculate the area formed by p1,p2,p3.
    // If the area is 0, then the points are collinear.
    float area = 0.5 * abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
    return area == 0;
}

__device__ __inline__
int points_are_equal(const float2 p1, const float2 p2) {
    return RESOLUTION >= hypotf(p2.x - p1.x, p2.y - p1.y);
}

__device__ __inline__
int point_on_line(const float2 p_test, const float2 p1, const float2 p2) {
    /**
     * Determine if a point is part of a line segment.
     * Distance from line->point must be <= RESOLUTION along the
     * norm of the line.
     *
     * @param p_test point to test
     * @param p1 end point of the line segment
     * @param p2 end point of the line segment
     * @return point is on line? 1 : 0
     */

    if (!isCollinear(p_test, p1, p2))
        return 0;


    return 1 * ( ((p_test.x <= fmaxf(p1.x, p2.x)) && (fmaxf(p1.x, p2.x) >= p_test.x)) &&
                  (p_test.y <= fmaxf(p1.y, p2.y)) && (fmaxf(p1.y, p2.y) >= p_test.y));
}

__device__ __inline__
int point_is_inside_triangle(const float2 p, const float2* polygon) {
    float2 a = polygon[0];
    float2 b = polygon[1];
    float2 c = polygon[2];

    float alpha = ((b.y - c.y)*(p.x - c.x) + (c.x - b.x)*(p.y - c.y)) /
                  ((b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y));

    float beta  = ((c.y - a.y)*(p.x - c.x) + (a.x - c.x)*(p.y - c.y)) /
                  ((b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y));

    float gamma = 1.0f - alpha - beta;

    return (alpha >= 0 and beta >= 0 and gamma >= 0) ? 1 : 0;
}

__global__
void CalculateWindingNumber2D(float *x_vals, float *y_vals, const float2 *polygon, size_t polygon_size, size_t count_points, int *results)
{
    // --- Index
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    float2 test_point;

    // --- Load Polygon into shared memory
    __shared__ float2 shared_polygon[100];
    const auto block = this_thread_block();

    if (block.thread_rank() < polygon_size)
        shared_polygon[threadIdx.x] = polygon[threadIdx.x];
    __syncthreads();


    // --- Bounds Check
    if (tid >= count_points)
        return;

    test_point.x = x_vals[tid];
    test_point.y = y_vals[tid];

    if (polygon_size == 1)
        results[tid] = points_are_equal(test_point, shared_polygon[0]);

    else if (polygon_size == 2)
        results[tid] = point_on_line(test_point, shared_polygon[0], shared_polygon[1]);

    else if (polygon_size == 3)
        results[tid] = point_is_inside_triangle(test_point, shared_polygon);

    else
    {
        int wn = 0;
        for (size_t i=0; i<polygon_size-1; i++) {
            float2 p0 = shared_polygon[i];
            float2 p1 = shared_polygon[i+1];

            // --- Upward Crossing
            if (p0.y <= test_point.y) {
                if (p1.y > test_point.y)
                    if (is_left(p0, p1, test_point) > 0)
                        ++wn;
            }

            // --- Downward Crossing
            else {
                if (p1.y <= test_point.y)
                    if (is_left(p0, p1, test_point) < 0)
                        --wn;
            }
        }

        results[tid] = wn;
    }
}

using namespace poly;

float GPU_Winding_Number_Solver::CalculateWindingNumber2D(float *x_vals, float *y_vals, int* results, const Polygon &polygon, size_t count_points)
{
    // --- Copy Polygon to Device
    float2 h_polygon[polygon.size()];
    for (int i=0; i<polygon.size(); i++)
        h_polygon[i] = {polygon.x_vec_[i], polygon.y_vec_[i]};

    float2 *d_polygon = __gpuMalloc(polygon.size());
    __gpuMemcpy(d_polygon, h_polygon, polygon.size());

    // --- Kernel Parameters
    dim3 block(512);
    dim3 grid;
    grid.x = (count_points / block.x) + 1 * ((count_points % block.x) != 0);

    __logDebug("Kernel Dims = Block(%d, %d, %d) Grid(%d, %d, %d)", block.x, block.y, block.z, grid.x, grid.y, grid.z);

    // --- Kernel Timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time_ms;

    // --- Run the Kernel
    cudaEventRecord(start);
    ::CalculateWindingNumber2D<<<grid, block>>>(x_vals, y_vals, d_polygon, polygon.size(), count_points, results);

    // --- Kernel Timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    __gpuErrCheck();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    fflush(stdout);
    return elapsed_time_ms * 1000; // microseconds
}


