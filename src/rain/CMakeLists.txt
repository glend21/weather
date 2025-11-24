cmake_minimum_required(VERSION 2.8)
project( RainProject )

find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( rain main.cpp ssim.cpp opticalflow.cpp farneback_flow.cpp tvl1_flow.cpp )
target_link_libraries( rain PRIVATE ${OpenCV_LIBS} Threads::Threads )
