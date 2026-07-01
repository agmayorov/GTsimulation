# Setup the project
cmake_minimum_required(VERSION 3.16)
project(DecayGenerator)

# Find Geant4 package, activating all available UI and Vis drivers by default
find_package(Geant4 REQUIRED ui_all vis_all)

# Locate sources and headers for this project
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Visualization option
option(USE_VISUALIZATION "Use visualization code" OFF)
if (USE_VISUALIZATION)
  add_definitions(-DUSE_VISUALIZATION)
  file(COPY ${PROJECT_SOURCE_DIR}/../vis.mac DESTINATION ${PROJECT_BINARY_DIR})
endif()

# Add the executable, use our local headers, and link it to the Geant4 libraries
add_executable(DecayGenerator DecayGenerator.cc ${sources} ${headers})
target_include_directories(DecayGenerator PRIVATE include)
target_link_libraries(DecayGenerator ${Geant4_LIBRARIES})

# Copy executable 
add_custom_command(
  TARGET DecayGenerator
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy $<TARGET_FILE:DecayGenerator> ${PROJECT_SOURCE_DIR}/../..
)
