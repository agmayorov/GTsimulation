# Setup the project
cmake_minimum_required(VERSION 3.16)
project(MatterLayer)

# Find Geant4 package
find_package(Geant4 REQUIRED)

# Locate sources and headers for this project
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Find pybind11 package
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED HINTS ${Python_SITELIB})

# Create Python extension module
pybind11_add_module(MatterLayer MatterLayer.cc ${sources} ${headers})
target_include_directories(MatterLayer PRIVATE include)
target_link_libraries(MatterLayer PRIVATE ${Geant4_LIBRARIES})
set_target_properties(MatterLayer PROPERTIES OUTPUT_NAME "matter_layer")

# # Copy executable
# add_custom_command(
#   TARGET MatterLayer
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND}
#   ARGS -E copy $<TARGET_FILE:MatterLayer> ${PROJECT_SOURCE_DIR}/../..
# )
