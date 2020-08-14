cmake_policy(SET CMP0074 NEW)

cmake_minimum_required(VERSION 2.8)
project( FeatureDetect )

# set( CMAKE_MODULE_PATH "${HOME}/lib64/;${CMAKE_MODULE_PATH}")
find_package( OpenCV REQUIRED )
message( ${OpenCV_FOUND} )
message( ${OpenCV_INCLUDE_DIRS} )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( feat1 feat1.cpp )
target_link_libraries( feat1 ${OpenCV_LIBS} )

add_executable( feat2 feat2.cpp )
target_link_libraries( feat2 ${OpenCV_LIBS} )
