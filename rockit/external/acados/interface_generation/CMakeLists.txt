cmake_minimum_required(VERSION 3.0.0)
project(interface_usage VERSION 0.1.0)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")

set(MODEL_NAME "rockit_model" CACHE STRING "Name of the model")

add_subdirectory(acados)

# Get all c files in directory rockit_model_model
file(GLOB_RECURSE ROCKIT_MODEL_MODEL_SRC ${MODEL_NAME}_model/*.c)
file(GLOB_RECURSE ROCKIT_MODEL_COST_SRC ${MODEL_NAME}_cost/*.c)
file(GLOB_RECURSE ROCKIT_MODEL_CONSTRAINTS_SRC ${MODEL_NAME}_constraints/*.c)



add_library(acados_driver SHARED acados_driver.c
  ${ROCKIT_MODEL_MODEL_SRC} ${ROCKIT_MODEL_COST_SRC} ${ROCKIT_MODEL_CONSTRAINTS_SRC}
  acados_solver_${MODEL_NAME}.c
  rockit_config.h
  )
set_property(SOURCE acados_driver.c APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/after_init.c.in)
target_link_libraries(acados_driver acados)

INSTALL(TARGETS acados_driver
  LIBRARY DESTINATION ${LIBRARY_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIBRARY_INSTALL_DIR}
  INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
  RUNTIME DESTINATION ${RUNTIME_INSTALL_DIR}
  )