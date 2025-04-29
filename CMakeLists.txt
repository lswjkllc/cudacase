cmake_minimum_required(VERSION 3.10)
project(cudacase LANGUAGES C CXX CUDA)

# set(CMAKE_C_COMPILER /usr/bin/gcc)
# set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cudacase 
    main.cu
    csrc/host/gemm.cc
    csrc/device/gemm/globalmem_2dmm.cu
    csrc/device/gemm/sharedmem_2dmm.cu
)
target_include_directories(cudacase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
