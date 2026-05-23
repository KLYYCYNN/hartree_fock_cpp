cmake_minimum_required(VERSION 3.18)

project(hf_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Your own headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Eigen and nlohmann/json are commonly found by CMake
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

# Header-only rapidcsv: assume you put rapidcsv.h in external/rapidcsv/
include_directories(${CMAKE_SOURCE_DIR}/external/rapidcsv)

# CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(test 
	tests/test.cpp
	cuda/eri.cu
)

target_link_libraries(test
    Eigen3::Eigen
    CUDA::cudart
)
