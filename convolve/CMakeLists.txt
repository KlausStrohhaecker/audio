# input variables: FIRMWARE_DIRNAME

cmake_minimum_required(VERSION 3.2)

project(convolve)

find_package(OpenMP)

IF(OPENMP_FOUND)
    message(" -> OpenMP found")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -s -O3 -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s -O3 -Wall -Wextra -Werror")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -Og -Wall -Wextra -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -Wall -Wextra -Werror")

include_directories(src ../src)

add_executable(convolve src/main.c src/convolve.c)

