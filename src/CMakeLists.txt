cmake_minimum_required(VERSION 2.8)
project( RainProject )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( rain main.cpp )
target_link_libraries( rain ${OpenCV_LIBS} )
