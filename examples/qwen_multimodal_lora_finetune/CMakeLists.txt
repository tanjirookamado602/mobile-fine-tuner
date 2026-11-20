cmake_minimum_required(VERSION 3.16)
project(QwenMultimodalLoRA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include cstdint -include string")

# Shared operator library root
get_filename_component(OPERATORS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../operator" ABSOLUTE)
set(BUILD_TESTS OFF CACHE BOOL "Disable operator tests in app builds" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable operator examples in app builds" FORCE)
add_subdirectory("${OPERATORS_ROOT}" operators_build)

add_executable(train_multimodal
    src/main.cpp
    src/qwen_multimodal_model.cpp
    src/multimodal_trainer.cpp
    src/multimodal_dataset.cpp
)
target_link_libraries(train_multimodal PRIVATE operators)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(train_multimodal PRIVATE ${ACCELERATE_LIBRARY})
    endif()
endif()
