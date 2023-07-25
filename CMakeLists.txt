cmake_minimum_required(VERSION 3.24)

# -----------------------------------
#             CUDA OPTIONS
# -----------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 61)

# --- Project
project(winding_number_homework LANGUAGES CXX CUDA)
include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        include
)


# --- Debug Options
add_compile_definitions("$<$<CONFIG:DEBUG>:__DEBUG__>")


# --- Common
add_library(common_lib OBJECT src/common.cu)

# --- Logging
add_library(logging_lib OBJECT src/logging.cu)


include(GoogleTest)
enable_testing()

# --- SMFL Visualization Library
find_package(SFML REQUIRED COMPONENTS system window graphics)
set(VIS_LIBRARIES
        sfml-system
        sfml-window
        sfml-graphics
)

# the guts of the library that computes winding number
set(WINDING_NUMBER_INC
        include/poly_io.hpp
        include/winding.hpp
        include/visualizer.h
        include/poly_creator.hpp
        include/gpu_winding_number.cuh
)

set(WINDING_NUMBER_SRC
        src/poly_io.cpp
        src/visualizer.cpp
        src/better_winding.cpp
        src/poly_creator.cpp
        src/gpu_winding_number.cu
)

add_library(winding_lib STATIC ${WINDING_NUMBER_SRC} ${WINDING_NUMBER_INC})
target_include_directories(winding_lib PUBLIC include)
target_link_libraries(winding_lib PUBLIC logging_lib common_lib)

# a main that is callable from a console
set(WINDING_NUMBER_MAIN
  src/main.cpp
)

add_executable(winding_number ${WINDING_NUMBER_MAIN})
target_link_libraries(winding_number PRIVATE winding_lib ${VIS_LIBRARIES})

# unit tests for the winding number homework problem
set(GTEST ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest)
set(GTEST_SRC_DIR ${GTEST}/src)
set(GTEST_INC_DIR ${GTEST}/include)

set(WINDING_NUMBER_TEST_SRC
  test/winding_test.cpp
  test/poly_io_test.cpp
  test/testmain.cpp
  ${GTEST_SRC_DIR}/gtest-all.cc
)

add_executable(winding_number_test ${WINDING_NUMBER_TEST_SRC})

target_include_directories(winding_number_test PRIVATE ${GTEST_INC_DIR} ${GTEST})

if (WIN32)
  target_link_libraries(winding_number_test PRIVATE winding_lib)
elseif(UNIX)
  target_link_libraries(winding_number_test PRIVATE winding_lib pthread)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test/polygons.txt ${CMAKE_CURRENT_BINARY_DIR}/polygons.txt COPYONLY)

gtest_discover_tests(winding_number_test)



