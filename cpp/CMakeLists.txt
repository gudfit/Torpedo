cmake_minimum_required(VERSION 3.20)
project(torpedocode_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

find_package(Torch REQUIRED)

add_library(torpedocode_kernels SHARED
    src/extension.cpp
    src/tpp_loss.cpp
    src/lob_kernels.cu
)

target_include_directories(torpedocode_kernels
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(torpedocode_kernels
    PUBLIC
        ${TORCH_LIBRARIES}
)

set_target_properties(torpedocode_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
