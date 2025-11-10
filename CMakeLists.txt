cmake_minimum_required(VERSION 3.10)
project(Ripples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
FetchContent_MakeAvailable(glfw)

add_executable(ripples src/main.cpp)

target_link_libraries(ripples PRIVATE glfw)
