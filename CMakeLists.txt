cmake_minimum_required(VERSION 3.15)
project(ibbv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src)

add_compile_options(-march=native)
add_library(ibbv ${CMAKE_CURRENT_LIST_DIR}/include/IndexedBlockBitVector.h)
target_include_directories(ibbv PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
)
set_target_properties(ibbv PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
