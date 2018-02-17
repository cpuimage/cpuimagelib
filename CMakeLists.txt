cmake_minimum_required(VERSION 3.9)
project(demo)

set(CMAKE_CXX_STANDARD 11)

add_executable(demo
        cpuimage.c
        demo.c )
