cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(lib_olm)

# set default build type
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE RelWithDebInfo)
# endif()
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-deprecated-declarations")

set(BUILD_SHARED_LIBS ON)

# file(GLOB INCLUDE_FILES "include/*.h" "kdtree/*.h")
# file(GLOB SRC_FILES "src/lro_rrt_server.cpp" "kdtree/*.c")

# add_subdirectory(lib_convex_decomp decomp_util)

find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED COMPONENTS common octree filters)
include_directories(
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    include
)

# add_library(${PROJECT_NAME} STATIC ${SRC_FILES} ${INCLUDE_FILES})
add_library(${PROJECT_NAME}
    src/mapper.cpp
)

target_include_directories (${PROJECT_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/include
    lib_convex_decomp/include
)