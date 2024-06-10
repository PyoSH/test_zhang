cmake_minimum_required(VERSION 3.0.2)
project(test_zhang)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(eigen_DIR "D:/eigen3")
find_package(eigen3 REQUIRED)

#set(OpenCV_DIR "D:/opencv")
find_package(OpenCV REQUIRED)

include_directories(
    src/
    ${eigen_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(test1 src/test1.cpp)
target_link_libraries(test1 ${eigen_LIBS})
target_link_libraries(test1 ${OpenCV_LIBS})


