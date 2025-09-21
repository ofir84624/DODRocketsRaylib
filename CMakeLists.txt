cmake_minimum_required(VERSION 3.15)
project(DODRocketsRaylib CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EXISTS ${CMAKE_BINARY_DIR}/cmake/CPM.cmake)
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
       ${CMAKE_BINARY_DIR}/cmake/CPM.cmake)
endif()
include(${CMAKE_BINARY_DIR}/cmake/CPM.cmake)

CPMAddPackage("gh:raysan5/raylib#5.5")
CPMAddPackage("gh:ocornut/imgui#v1.92.3")
CPMAddPackage("gh:raylib-extras/rlImGui#main")

if(NOT TARGET imgui)
  set(IMGUI_SOURCES
      ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
      ${imgui_SOURCE_DIR}/imgui_tables.cpp
      ${imgui_SOURCE_DIR}/imgui_widgets.cpp)

  add_library(imgui STATIC ${IMGUI_SOURCES})
  target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
  target_compile_features(imgui PUBLIC cxx_std_20)
endif()

if(NOT TARGET rlImGui)
  add_library(rlImGui STATIC ${rlImGui_SOURCE_DIR}/rlImGui.cpp)
  target_include_directories(rlImGui PUBLIC ${rlImGui_SOURCE_DIR}
                                            ${imgui_SOURCE_DIR})
  target_compile_features(rlImGui PUBLIC cxx_std_20)
  target_link_libraries(rlImGui PUBLIC imgui raylib)
endif()

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE raylib imgui rlImGui)

target_include_directories(${PROJECT_NAME} PRIVATE ${imgui_SOURCE_DIR}
                                                   ${rlImGui_SOURCE_DIR})
