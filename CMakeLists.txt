cmake_minimum_required(VERSION 3.15)
project(btree_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get pybind11 include directory
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()

# Get Python extension suffix
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')"
    OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Found pybind11: ${PYBIND11_INCLUDE_DIR}")
message(STATUS "Python includes: ${Python3_INCLUDE_DIRS}")
message(STATUS "Module extension: ${PYTHON_MODULE_EXTENSION}")

# ==============================================================================
# B-Tree Module
# ==============================================================================
add_library(btree_cpp MODULE
    src/indexes/cpp/btree_cpp.cpp
    src/indexes/cpp/bindings.cpp
)

target_include_directories(btree_cpp PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src/indexes/cpp/include
)

target_link_libraries(btree_cpp PRIVATE
    ${Python3_LIBRARIES}
)

set_target_properties(btree_cpp PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "btree_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/indexes/cpp"
)

# ==============================================================================
# Linear Model Module
# ==============================================================================
add_library(linear_model_cpp MODULE
    src/indexes/cpp/linear_model_cpp.cpp
    src/indexes/cpp/bindings_linear.cpp
)

target_include_directories(linear_model_cpp PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src/indexes/cpp/include
)

target_link_libraries(linear_model_cpp PRIVATE
    ${Python3_LIBRARIES}
)

set_target_properties(linear_model_cpp PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "linear_model_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/indexes/cpp"
)

# ==============================================================================
# RMI Module
# ==============================================================================
add_library(rmi_cpp MODULE
    src/indexes/cpp/rmi_cpp.cpp
    src/indexes/cpp/bindings_rmi.cpp
)

target_include_directories(rmi_cpp PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src/indexes/cpp/include
)

target_link_libraries(rmi_cpp PRIVATE
    ${Python3_LIBRARIES}
)

set_target_properties(rmi_cpp PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "rmi_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/indexes/cpp"
)

message(STATUS "All modules will be built to: ${PROJECT_SOURCE_DIR}/src/indexes/cpp")