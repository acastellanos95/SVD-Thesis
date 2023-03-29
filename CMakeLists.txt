cmake_minimum_required(VERSION 3.22)


set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER_ID NVIDIA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_STANDARD 17)
add_compile_options(-G)

project(SVD_Thesis LANGUAGES CUDA CXX)

#find_package(MKL CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
IF (OPENMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS} --std=c++17")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

add_executable(SVD_Thesis src/main.cu lib/HouseholderMethods.cuh lib/HouseholderMethods.cu lib/Matrix.cuh lib/global.cuh lib/JacobiMethods.cuh lib/JacobiMethods.cu lib/Utils.cu lib/Utils.cuh)

#target_compile_options(SVD_Thesis PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
#target_include_directories(SVD_Thesis PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
#set_target_properties(SVD_Thesis PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(SVD_Thesis cublas curand $<LINK_ONLY:MKL::MKL>)

#target_compile_options(SVD_Thesis PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
#target_include_directories(SVD_Thesis PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
set_target_properties(SVD_Thesis PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(SVD_Thesis cublas curand)
