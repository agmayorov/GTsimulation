# gtsimulation/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(GTsimulation LANGUAGES CXX)

set(PYTHON_PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/gtsimulation)
set(INTERACTIONS_SUBMODULE_DIR ${PYTHON_PACKAGE_SOURCE_DIR}/Interaction)

# --- Option for building Geant4-dependent components ---
option(BUILD_GEANT4_COMPONENTS "Build components that require Geant4" ON)

# Variable for storing the build status of Geant4 components
set(GEANT4_COMPONENTS_WERE_BUILT FALSE CACHE INTERNAL "Were Geant4 components successfully built")

if(BUILD_GEANT4_COMPONENTS)
    message(STATUS "BUILD_GEANT4_COMPONENTS is ON. Attempting to find Geant4...")
    # Looking for Geant4. QUIET so that there is no error if not found.
    find_package(Geant4 QUIET)

    if(Geant4_FOUND)
        message(STATUS "Geant4 found at ${Geant4_DIR}. Building Geant4-dependent components.")
        set(GEANT4_COMPONENTS_WERE_BUILT TRUE)
        # --- Adding and installing your C++ subprojects ---
        add_subdirectory(${INTERACTIONS_SUBMODULE_DIR}/G4source/MatterLayer ${CMAKE_BINARY_DIR}/MatterLayer_build)
        add_subdirectory(${INTERACTIONS_SUBMODULE_DIR}/G4source/DecayGenerator ${CMAKE_BINARY_DIR}/DecayGenerator_build)
        add_subdirectory(${INTERACTIONS_SUBMODULE_DIR}/G4source/Atmosphere ${CMAKE_BINARY_DIR}/Atmosphere_build)
        install(
            TARGETS MatterLayer DecayGenerator Atmosphere
            RUNTIME DESTINATION ${SKBUILD_PROJECT_NAME}/Interaction
        )
    else()
        message(WARNING "Geant4 not found, although BUILD_GEANT4_COMPONENTS was ON. Geant4-dependent components will NOT be built.")
    endif()
else()
    message(STATUS "BUILD_GEANT4_COMPONENTS is OFF. Geant4-dependent components will NOT be built.")
endif()

# --- Generating a Python build configuration file ---
set(GENERATED_PYTHON_CONFIG_FILE ${INTERACTIONS_SUBMODULE_DIR}/_build_config.py)

if(GEANT4_COMPONENTS_WERE_BUILT)
    set(GEANT4_COMPONENTS_WERE_BUILT_BOOL True)
    if(DEFINED Geant4_INSTALL_PREFIX)
        set(GEANT4_INSTALL_PREFIX_STRING "${Geant4_INSTALL_PREFIX}")
    elseif(DEFINED Geant4_PREFIX_PATH)
        set(GEANT4_INSTALL_PREFIX_STRING "${Geant4_PREFIX_PATH}")
    else()
        get_filename_component(G4_CMAKE_DIR ${Geant4_DIR} DIRECTORY)
        get_filename_component(G4_LIB_DIR ${G4_CMAKE_DIR} DIRECTORY)
        get_filename_component(GEANT4_INSTALL_PREFIX_STRING ${G4_LIB_DIR} DIRECTORY)
        message(STATUS "Geant4_INSTALL_PREFIX is not defined, use calculated prefix: ${GEANT4_INSTALL_PREFIX_STRING}")
    endif()
else()
    set(GEANT4_COMPONENTS_WERE_BUILT_BOOL False)
    set(GEANT4_INSTALL_PREFIX_STRING "")
endif()

configure_file(
    ${PYTHON_PACKAGE_SOURCE_DIR}/Interaction/_build_config.py.in
    ${GENERATED_PYTHON_CONFIG_FILE}
    @ONLY
)
message(STATUS "Build configuration file generated: ${GENERATED_PYTHON_CONFIG_FILE}")
