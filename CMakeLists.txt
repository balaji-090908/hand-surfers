cmake_minimum_required(VERSION 3.10)
project(HandSurfers CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(hand_surfers main.cpp)
target_include_directories(hand_surfers PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hand_surfers PRIVATE ${OpenCV_LIBS})

if(APPLE)
  target_link_libraries(hand_surfers PRIVATE "-framework ApplicationServices")
endif()
