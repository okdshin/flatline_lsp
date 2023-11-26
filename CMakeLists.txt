cmake_minimum_required(VERSION 3.17)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(flatline C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF)

include(FetchContent)

FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/okdshin/llama.cpp.git
  GIT_TAG add_pfnet_plamo_13b
)

FetchContent_Declare(
  asio
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG asio-1-28-1
)

set(JSONCPP_WITH_TESTS OFF)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF)
set(JSONCPP_WITH_TESTS OFF)
FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG 1.9.5
)

FetchContent_Declare(
  structopt
  URL "https://github.com/p-ranav/structopt/archive/refs/tags/v0.1.3.zip")

set(SPDLOG_BUILD_SHARED OFF)
FetchContent_Declare(
  spdlog URL "https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.zip")

FetchContent_MakeAvailable(llama_cpp asio jsoncpp structopt spdlog)

add_library(asio INTERFACE)
target_include_directories(asio INTERFACE "${asio_SOURCE_DIR}/asio/include")

FetchContent_Declare(
  cinatra
  GIT_REPOSITORY https://github.com/okdshin/cinatra.git
  GIT_TAG v0.8.0_g++10
)
FetchContent_Populate(cinatra)
add_library(cinatra INTERFACE)
target_include_directories(cinatra INTERFACE "${cinatra_SOURCE_DIR}/include")
find_package(Threads REQUIRED)
target_link_libraries(cinatra INTERFACE Threads::Threads asio)
target_compile_definitions(cinatra INTERFACE -DASIO_STANDALONE)

add_executable(${PROJECT_NAME}-server
               ${CMAKE_CURRENT_SOURCE_DIR}/flatline_server.cpp)
set_target_properties(${PROJECT_NAME}-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
target_link_libraries(${PROJECT_NAME}-server PRIVATE llama ggml cinatra jsoncpp_static structopt spdlog::spdlog)
add_custom_command(
    TARGET ${PROJECT_NAME}-server
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/flatline.LICENSE.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${asio_SOURCE_DIR}/asio/LICENSE_1_0.txt $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/asio.LICENSE.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${cinatra_SOURCE_DIR}/LICENSE $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/cinatra.LICENSE.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${jsoncpp_SOURCE_DIR}/LICENSE $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/jsoncpp.LICENSE.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${llama_cpp_SOURCE_DIR}/LICENSE $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/llama_cpp.LICENSE.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${spdlog_SOURCE_DIR}/LICENSE $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/spdlog.LICENSE.txt
    # structopt
    COMMAND ${CMAKE_COMMAND} -E copy ${structopt_SOURCE_DIR}/LICENSE $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/structopt.LICENSE.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${structopt_SOURCE_DIR}/LICENSE.magic_enum $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/structopt.magic_enum.LICENSE.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${structopt_SOURCE_DIR}/LICENSE.visit_struct $<TARGET_FILE_DIR:${PROJECT_NAME}-server>/structopt.visit_struct.LICENSE.txt
)
