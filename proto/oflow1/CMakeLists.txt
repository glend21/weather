cmake_policy(SET CMP0074 NEW)
cmake_minimum_required(VERSION 2.8)

project( OpticalFlow )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( optflow optflow1.cpp )
target_link_libraries( optflow ${OpenCV_LIBS} )

add_executable( fback fback.cpp )
target_link_libraries( fback ${OpenCV_LIBS} )
