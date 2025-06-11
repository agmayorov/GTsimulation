# gtsimulation/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

project(GTsimulation NONE)
set(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/gtsimulation)

# --- Option for building Geant4-dependent components ---
option(BUILD_GEANT4_COMPONENTS "Build components that require Geant4" ON)

# Variable for storing the build status of Geant4 components
set(GEANT4_COMPONENTS_WERE_BUILT FALSE CACHE INTERNAL "Were Geant4 components successfully built")

if(BUILD_GEANT4_COMPONENTS)
    message(STATUS "BUILD_GEANT4_COMPONENTS is ON. Attempting to find Geant4...")
    # Looking for Geant4. QUIET so that there is no error if not found.
    enable_language(CXX OPTIONAL)
    find_package(Geant4 QUIET)

    if(Geant4_FOUND)
        message(STATUS "Geant4 found at ${Geant4_DIR}. Building Geant4-dependent components.")
        set(GEANT4_COMPONENTS_WERE_BUILT TRUE)
        # --- Adding and installing your C++ subprojects ---
        add_subdirectory(${PACKAGE_SOURCE_DIR}/Interaction/G4source/MatterLayer ${CMAKE_BINARY_DIR}/build_ML)
        add_subdirectory(${PACKAGE_SOURCE_DIR}/Interaction/G4source/DecayGenerator ${CMAKE_BINARY_DIR}/build_DG)
        add_subdirectory(${PACKAGE_SOURCE_DIR}/Interaction/G4source/Atmosphere ${CMAKE_BINARY_DIR}/build_A)
        install(
            TARGETS MatterLayer DecayGenerator Atmosphere
            DESTINATION ${SKBUILD_PROJECT_NAME}/Interaction
        )
    else()
        message(WARNING "Geant4 not found, although BUILD_GEANT4_COMPONENTS was ON. Geant4-dependent components will NOT be built.")
    endif()
else()
    message(STATUS "BUILD_GEANT4_COMPONENTS is OFF. Geant4-dependent components will NOT be built.")
endif()

# --- Generating a Python build configuration file ---
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
    ${PACKAGE_SOURCE_DIR}/Interaction/_build_config.py.in
    ${PACKAGE_SOURCE_DIR}/Interaction/_build_config.py
    @ONLY
)
message(STATUS "Build configuration file generated: ${PACKAGE_SOURCE_DIR}/Interaction/_build_config.py")

install(FILES
  ${PACKAGE_SOURCE_DIR}/Interaction/_build_config.py
  DESTINATION ${SKBUILD_PROJECT_NAME}/Interaction
)
