cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source build is not supported")
endif()

set(PROJECT_NAME cpplm)

project(${PROJECT_NAME} VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG main
	FIND_PACKAGE_ARGS NAMES googletest
)
FetchContent_MakeAvailable(googletest)

# uncomment for testing
# enable_testing()
