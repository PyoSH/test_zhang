	cmake_minimum_required(VERSION 3.0.2)
project(test_zhang)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

## IF U NEED TO SET LIBS
#set(EIGEN3_INCLUDE_DIR "D:/eigen3")
#set(Eigen3_LIB "D:/eigen3")
#set(OPENCV_DIR "D:/opencv/build/include")
#set(OPENCV_LIB "D:/opencv/build/include")
#include_directories(${EIGEN3_INCLUDE_DIR})
#include_directories(${OPENCV_DIR})

find_package(Eigen3)
find_package(OpenCV REQUIRED)

add_executable(test1 src/test1.cpp)
target_link_libraries(test1 ${OpenCV_LIBS})