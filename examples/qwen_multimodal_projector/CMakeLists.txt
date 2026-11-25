cmake_minimum_required(VERSION 3.16)
project(qwen_multimodal_projector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT ANDROID AND EXISTS "/home/assassinaj602/openblas_install/lib/libopenblas.so")
    list(APPEND CMAKE_PREFIX_PATH "/home/assassinaj602/openblas_install")
    set(BLAS_LIBRARIES "/home/assassinaj602/openblas_install/lib/libopenblas.so" CACHE FILEPATH "OpenBLAS lib" FORCE)
    set(USE_BLAS ON CACHE BOOL "Enable BLAS on host" FORCE)
    include_directories(/home/assassinaj602/openblas_install/include)
endif()

# Shared operator library root
get_filename_component(OPERATORS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../operator" ABSOLUTE)
set(BUILD_TESTS OFF CACHE BOOL "Disable operator tests in app builds" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable operator examples in app builds" FORCE)
add_subdirectory("${OPERATORS_ROOT}" operators_build)

add_executable(train_projector train_projector.cpp)
add_executable(profile_memory profile_memory.cpp)
add_executable(parity_5step_runner parity_5step_runner.cpp)

foreach(tgt train_projector profile_memory parity_5step_runner)
    target_include_directories(${tgt} PRIVATE
        ${OPERATORS_ROOT}
        ${OPERATORS_ROOT}/finetune_ops
    )
    target_link_libraries(${tgt} PRIVATE operators)
    if(ANDROID)
        target_link_libraries(${tgt} PRIVATE log)
    elseif(EXISTS "/home/assassinaj602/openblas_install/lib/libopenblas.so")
        target_include_directories(${tgt} PRIVATE /home/assassinaj602/openblas_install/include)
        target_link_libraries(${tgt} PRIVATE /home/assassinaj602/openblas_install/lib/libopenblas.so pthread dl)
    endif()
endforeach()
