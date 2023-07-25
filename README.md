# Winding Number Homework Problem

**For more information on the process and implementation:  
Go to [Project Writeup](Project_Writeup.md)**


## Overview
Take home assessment for Dyndrite. Original repository is located here:
https://bitbucket.org/dynall/winding_number_homework/src/master/

## Objectives

- Implement the winding number algorithm by subclassing the `IWindingNumberAlgorithm` interface class.
- Make the code that reads in the input text file a bit more resilient to bad input.

## Getting Started

> :point_right: The solution has been implemented on the following system.  
> It will be relatively portable to other systems, but has not yet been tested and validated for portability.

- Fedora 38
- CMake 3.26.4
- g++ 13.1.1
- C++ 23
- nvcc 12.1
- CUDA 12.1
- NVidia Driver 535.54.03
- GTX 1060
- Intel 6700k

### Install SFML Library
> :exclamation:  SFML Library  
> An additional library (SFML) was added to enable visualization of the polygons and points.  
> The SFML library can be downloaded and installed [here](https://www.sfml-dev.org/download.php).  
> Or through the terminal :

```shell
# Fedora
sudo dnf install SFML-devel

# Ubuntu
sudo apt-get install libsfml-dev
```

### Update CUDA Architecture
In [CMakeLists.txt](CMakeLists.txt) update the following line to compile for the correct CUDA architecture.  
```cmake
# For GTX 10 series cards (pascal sm_61)
set(CMAKE_CUDA_ARCHITECTURES 61)
```
[Matching CUDA arch and CUDA gencode](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

| Fermi | Kepler | Maxwell | Pascal | Volta          | Turing | Ampere       | Ada (Lovelace) | Hopper        |
|-------|--------|---------|--------|----------------|--------|--------------|----------------|---------------|
| sm_20 | sm_30  | sm_50   | sm_60  | sm_70          | sm_75  | sm_80        | sm_89          | sm_90         |
|       | sm_35  | sm_52   | sm_61  | sm_72 (Xavier) |        | sm_86        |                | sm_90a (Thor) |
|       | sm_37  | sm_53   | sm_62  |                |        | sm_87 (Orin) |                |               |
# Directory

- **Common** -- macros and wrappers for cuda functions
  - [common.cuh](include/common.cuh)
  - [common.cu](src/common.cu)
- **Logging** -- C logging library with debug / release function variants
  - [logging.h](include/logging.h)
  - [loggin.cu](src/logging.cu)
- **Visualizer** -- display points and polygons to a window using SFML
  - [visualizer.h](include/visualizer.h)
  - [visualizer.cpp](src/visualizer.cpp)
- **Winding Number Algorithm**
  - [winding.hpp](include/winding.hpp)
  - [better_winding.cpp](src/better_winding.cpp)
  - [gpu_winding_number.cuh](include/gpu_winding_number.cuh)
  - [gpu_winding_number.cu](src/gpu_winding_number.cu)
  - [wingind.cpp](src/winding.cpp)
- **Polygons**
  - [poly_io.hpp](include/poly_io.hpp)
  - [poly_creator.hpp](include/poly_creator.hpp)
  - [poly_io.cpp](src/poly_io.cpp)
  - [polycreator.cpp](src/poly_creator.cpp)

> :point_right: The Visualizer class DOES NOT display non-simple polygons correctly
> but they are still correctly assessed in the winding-number algorithm.  
> In order to correctly display non-simple polygons they will need to be broken into
> component polygons which can be displayed independently.