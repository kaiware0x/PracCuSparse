cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CUDA_ARCHITECTURES 86) # 使用するGPUのCompute Capability
# set(CUDAToolkit_ROOT "Your-CUDA-Install-Dir") # PCに複数VerのCUDAがインストールされている場合は明示的に指定する.

project(PracCuSparse VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit)

# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 11)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()

add_executable(${PROJECT_NAME}  
    main.cu
    spmv_csr_example.hpp
    )

target_link_libraries(${PROJECT_NAME} PRIVATE 
    cudart
    cusparse
)

# set_target_properties(${PROJECT_NAME} PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
