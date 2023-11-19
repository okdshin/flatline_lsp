cmake_minimum_required(VERSION 3.17)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(local_llm C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Python COMPONENTS Interpreter Development)

include(FetchContent)

FetchContent_Declare(pybind11 URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip")
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/okdshin/llama.cpp.git
  GIT_TAG add_pfnet_plamo_13b
)
set(BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(pybind11 llama_cpp)


pybind11_add_module(infer infer.cpp)
target_link_libraries(infer PRIVATE llama ggml)
