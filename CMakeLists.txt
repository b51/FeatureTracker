cmake_minimum_required(VERSION 2.8.3)
project(FeatureTracker)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

#########################################################################
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC -march=native")
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fPIC")
endif(CMAKE_BUILD_TYPE STREQUAL "Release")
#########################################################################

find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  #CATKIN_DEPENDS
)

include_directories(
  include
  ${OPENCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(${PROJECT_NAME}
  src/main.cc
  src/FeatureDescriptor.cc
  src/ORBFeatureDetector.cc
)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  glog
  gflags
)
