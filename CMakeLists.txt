cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(OpenCLPractice VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(juliaset main.cpp)
target_link_libraries(juliaset ${OpenCL_LIBRARIES} ${OpenCV_LIBS})