cmake_minimum_required(VERSION 3.10)
project(qwen_multimodal_projector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(train_projector train_projector.cpp)

target_include_directories(train_projector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../operator
    ${CMAKE_CURRENT_SOURCE_DIR}/../../operator/finetune_ops
    /home/assassinaj602/openblas_install/include
)

target_link_libraries(train_projector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../operator/build/lib/liboperators.a
    /home/assassinaj602/openblas_install/lib/libopenblas.a
    pthread
    dl
)
