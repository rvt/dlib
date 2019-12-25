cmake_minimum_required(VERSION 3.3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_BUILD_TYPE Release)
add_definitions(-msse2)
add_definitions(-mavx)
add_definitions("-Wreturn-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -DUSE_SSE2_INSTRUCTIONS=ON")


project( DisplayImage )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( dlib REQUIRED )
include_directories( dlib::dlib )

add_executable( opencv main.cpp )
target_link_libraries( opencv ${OpenCV_LIBS} )

add_executable( dlib dlib.cpp )
target_link_libraries(dlib dlib::dlib ${OpenCV_LIBS})

