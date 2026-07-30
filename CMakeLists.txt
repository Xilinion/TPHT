cmake_minimum_required(VERSION 3.12)
project(tpht C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(tpht tpht.c)
target_include_directories(tpht PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(tpht_test tests/tpht_test.c)
target_link_libraries(tpht_test PRIVATE tpht)

enable_testing()
add_test(NAME tpht_test COMMAND tpht_test)
