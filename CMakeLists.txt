cmake_minimum_required(VERSION 2.8.3)
project(FeatureTracker)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

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

#find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

######## LibPytorch ########
#find_package(PythonLibs 3.7 REQUIRED)
#execute_process(COMMAND
#  python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
#  OUTPUT_VARIABLE
#  PYTHON_SITE_PACKAGES
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#)
#list(APPEND CMAKE_PREFIX_PATH ${PYTHON_SITE_PACKAGES}/torch/share/cmake)
#list(APPEND CMAKE_PREFIX_PATH "~/.local/lib/python3.6/site-packages/torch/share/cmake")
#find_package(Torch REQUIRED)

#catkin_package(
#  INCLUDE_DIRS include
#  #CATKIN_DEPENDS
#)

include_directories(
  include
  ${OPENCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

### Declare a C++ executable
#add_executable(${PROJECT_NAME}_orb
#  src/main_track_orb.cc
#  src/ORBFeatureDetector.cc
#  src/ORBFeatureMatcher.cc
#  src/ORBFeatureTracker.cc
#)
#
### Specify libraries to link a library or executable target against
#target_link_libraries(${PROJECT_NAME}_orb
#  ${OpenCV_LIBS}
#  glog
#  gflags
#)

## Declare a C++ executable
add_executable(${PROJECT_NAME}
  src/main_track.cc
  src/ORBFeatureDetector.cc
  src/ORBFeatureMatcher.cc
  src/ORBFeatureTracker.cc
  #  src/SuperPointFeatureDetector.cc
  #  src/SuperPointFeatureMatcher.cc
  #  src/SuperPointFeatureTracker.cc
)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME} PUBLIC
  OpenMP::OpenMP_CXX
  #${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  glog
  gflags
)
