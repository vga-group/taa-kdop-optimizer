cmake_minimum_required(VERSION 3.16)

project(KDopOptimizers LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glm REQUIRED)
find_package(OpenMP)

add_executable(sphere_optimizer sphere_optimizer.cc)
target_link_libraries(sphere_optimizer PUBLIC glm::glm)
target_compile_features(sphere_optimizer PUBLIC cxx_std_17)
set_property(TARGET sphere_optimizer PROPERTY CXX_STANDARD 17)
set_property(TARGET sphere_optimizer PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET sphere_optimizer PROPERTY CXX_EXTENSIONS OFF)

add_executable(image_optimizer image_optimizer.cc)
target_link_libraries(image_optimizer PUBLIC glm::glm)
if(OpenMP_CXX_FOUND)
    target_link_libraries(image_optimizer PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_features(image_optimizer PUBLIC cxx_std_17)
set_property(TARGET image_optimizer PROPERTY CXX_STANDARD 17)
set_property(TARGET image_optimizer PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET image_optimizer PROPERTY CXX_EXTENSIONS OFF)
