cmake_minimum_required(VERSION 3.18)

project(hf_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


include_directories(${CMAKE_SOURCE_DIR}/include)


find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/external/rapidcsv)

# CUDA
find_package(CUDAToolkit REQUIRED)

#-------------------------------------------

add_library(hf_core INTERFACE)

target_include_directories(hf_core INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/rapidcsv
    ${CMAKE_SOURCE_DIR}/cuda/include
)

target_link_libraries(hf_core INTERFACE
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    CUDA::cudart
)

#---------------------------------

add_library(vis INTERFACE)

target_include_directories(vis INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/rapidcsv
)

target_link_libraries(vis INTERFACE
    nlohmann_json::nlohmann_json
)

#----------------------------------------------

add_executable(test tests/test.cpp cuda/src/eri.cu)
target_link_libraries(test hf_core)

add_executable(H2O tests/H2O.cpp cuda/src/eri.cu)
target_link_libraries(H2O hf_core)

add_executable(2dslice tests/2dslice.cpp cuda/src/visual.cu)
target_link_libraries(2dslice hf_core)

add_executable(cube tests/cube.cpp cuda/src/visual.cu)
target_link_libraries(cube hf_core)

add_executable(test1 tests/test.cu)
target_link_libraries(test1 hf_core)

add_executable(test2 tests/test2.cpp cuda/src/eri.cu)
target_link_libraries(test2 hf_core)

add_executable(debug tests/eri_debug.cpp cuda/src/eri.cu)
target_link_libraries(debug hf_core)
#---------------------------------------------------
