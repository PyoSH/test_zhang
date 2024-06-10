cmake_minimum_required(VERSION 3.0.2)
project(test_zhang)


if(DEFINED ${eigen_DIR})
    find_package(eigen3)
endif()

if(NOT DEFINED ${eigen_DIR})
    set(eigen_DIR "D:/eigen3")
endif()

set(OpenCV_DIR "D:/opencv")
find_package(OpenCV REQUIRED)


include_directories(
    src/
    ${eigen_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(test1 src/test1.cpp)
target_link_libraries(test1 ${eigen_LIBS})
target_link_libraries(test1 ${OpenCV_LIBS})


