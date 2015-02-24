##
#  CMake script for the step-3 tutorial program:
##

# Set the name of the project and target:
# SET(TARGET "LaplaceBeltrami_1_hp_dofhandler_v820_copied_v5")
# SET(TARGET "LaplaceBeltrami_1_hp_dofhandler_v820_copied_v5_coupled")
# SET(TARGET "LaplaceBeltrami_1_hp_dofhandler_v820_copied_v5_coupled_new_rhs")
# new target:
SET(TARGET "LaplaceBeltrami_v6")

# Declare all source files the target consists of:
SET(TARGET_SRC
  ${TARGET}.cc

	/home/afonsoal/Documents/dealii_8_2_1_unzipped/dealii-8.2.1/my_programs/my_includes/cut_cell_integration.cpp
	/home/afonsoal/Documents/dealii_8_2_1_unzipped/dealii-8.2.1/my_programs/my_includes/NewCell.cpp
	/home/afonsoal/Documents/dealii_8_2_1_unzipped/dealii-8.2.1/my_programs/my_includes/NewMesh.cpp


  # You can specify additional files here!
  )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
