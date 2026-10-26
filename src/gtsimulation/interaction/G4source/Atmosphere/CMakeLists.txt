# Setup the project
cmake_minimum_required(VERSION 3.16)
project(Atmosphere)

# Find Geant4 package
find_package(Geant4 REQUIRED)

# Locate sources and headers for this project
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Find pybind11 package
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED HINTS ${Python_SITELIB})

# Create Python extension module
pybind11_add_module(Atmosphere Atmosphere.cc ${sources} ${headers})
target_include_directories(Atmosphere PRIVATE include)
target_link_libraries(Atmosphere PRIVATE ${Geant4_LIBRARIES} Python::Python)
set_target_properties(Atmosphere PROPERTIES
    OUTPUT_NAME "atmosphere"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
