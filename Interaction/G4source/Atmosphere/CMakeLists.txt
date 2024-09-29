# Setup the project
cmake_minimum_required(VERSION 3.16...3.27)
project(Atmosphere)

# Find Geant4 package, activating all available UI and Vis drivers by default
find_package(Geant4 REQUIRED ui_all vis_all)

# Setup Geant4 include directories and compile definitions
include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Locate sources and headers for this project
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Add the executable, and link it to the Geant4 libraries
add_executable(Atmosphere Atmosphere.cc ${sources} ${headers})
target_link_libraries(Atmosphere ${Geant4_LIBRARIES})

# For internal Geant4 use
add_custom_target(AS DEPENDS Atmosphere)

# Copy executable 
add_custom_command(TARGET Atmosphere
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                   ARGS -E copy $<TARGET_FILE:Atmosphere> ${PROJECT_SOURCE_DIR}/../..)
