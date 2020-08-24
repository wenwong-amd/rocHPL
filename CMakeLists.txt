# Modifications (c) 2019 Advanced Micro Devices, Inc.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     /opt/rocm/hip/cmake)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# rocHPL project
project(rochpl LANGUAGES CXX)

# Force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(HPL_DEBUG "Compile with modest debugging turned on" OFF)
option(HPL_DETAILED_DEBUG "Compile with voluminous debugging information turned on" OFF)
option(HPL_DETAILED_TIMING "Enable detail timers" OFF)
option(HPL_REFERENCE "Build reference mode" OFF)
option(BUILD_TEST "Build rocHPL single-node test" OFF)

# Optimization options
option(OPT_MEMMGMT "Build with memory management module" ON)
option(OPT_DEFRAG "Build with memory management defragmentation" ON)

# Dependencies
include(cmake/Dependencies.cmake)

# Setup version
rocm_setup_version(VERSION 0.7.3)

# AMD targets
set(AMDGPU_TARGETS gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target")

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure a header file to pass the rocALUTION version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp.in"
               "${PROJECT_BINARY_DIR}/include/Version.hpp"
)

# HPL HIP sources
set(rochpl_hip_source
  ComputeDotProduct.cpp
  ComputeSPMV.cpp
  ComputeSYMGS.cpp
  ComputeProlongation.cpp
  ComputeResidual.cpp
  ComputeRestriction.cpp
  ComputeWAXPBY.cpp
  ExchangeHalo.cpp
  GenerateCoarseProblem.cpp
  GenerateProblem.cpp
  init.cpp
  MultiColoring.cpp
  Permute.cpp
  SetupHalo.cpp
  SparseMatrix.cpp
  TestCG.cpp
)

# HPL sources
set(rochpl_source
  CG.cpp
  CG_ref.cpp
  CheckAspectRatio.cpp
  CheckProblem.cpp
  ComputeDotProduct_ref.cpp
  ComputeMG.cpp
  ComputeMG_ref.cpp
  ComputeOptimalShapeXYZ.cpp
  ComputeProlongation_ref.cpp
  ComputeRestriction_ref.cpp
  ComputeSPMV_ref.cpp
  ComputeSYMGS_ref.cpp
  ComputeWAXPBY_ref.cpp
  GenerateGeometry.cpp
  Memory.cpp
  MixedBaseCounter.cpp
  OptimizeProblem.cpp
  OutputFile.cpp
  ReadHpcgDat.cpp
  ReportResults.cpp
  TestNorms.cpp
  TestSymmetry.cpp
  WriteProblem.cpp
  YAML_Doc.cpp
  YAML_Element.cpp
  finalize.cpp
  mytimer.cpp
)

# Flag source files as hip source files
foreach(i ${rochpl_hip_source})
  set_source_files_properties(${i} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
endforeach()

# HIP flags workaround while target_compile_options does not work
list(APPEND HIP_HIPCC_FLAGS "-Wno-unused-command-line-argument -Wno-duplicate-decl-specifier")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND HIP_HIPCC_FLAGS "-g")
endif()


foreach(target ${AMDGPU_TARGETS})
  list(APPEND HIP_HIPCC_FLAGS "--amdgpu-target=${target}")
endforeach()

# Target executable
hip_add_executable(rochpl ${rochpl_source} main.cpp ${rochpl_hip_source})

# Target compile definitions
list(APPEND CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_HCC__")

# Support only for contiguous arrays
target_compile_definitions(rochpl PRIVATE HPCG_CONTIGUOUS_ARRAYS)

if(HPCG_DEBUG)
  target_compile_definitions(rochpl PRIVATE HPCG_DEBUG)
endif()

if(HPCG_DETAILED_DEBUG)
  target_compile_definitions(rochpl PRIVATE HPCG_DETAILED_DEBUG)
endif()

if(HPCG_DETAILED_TIMING)
  target_compile_definitions(rochpl PRIVATE HPCG_DETAILED_TIMING)
endif()

if(HPCG_REFERENCE)
  target_compile_definitions(rochpl PRIVATE HPCG_REFERENCE)
endif()

if(BUILD_TEST)
  target_compile_definitions(rochpl PRIVATE GOOGLE_TEST)
endif()

# Target include directories
target_include_directories(rochpl
                           PRIVATE
                           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                           $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>)

# MPI
if(HPCG_MPI)
  target_link_libraries(rochpl PRIVATE MPI::MPI_CXX libnuma::libnuma)
else()
  target_compile_definitions(rochpl PRIVATE HPCG_NO_MPI)
endif()

# OpenMP
if(HPCG_OPENMP)
  target_link_libraries(rochpl PRIVATE OpenMP::OpenMP_CXX)
else()
  target_compile_definitions(rochpl PRIVATE HPCG_NO_OPENMP)
endif()

# Target link libraries
target_link_libraries(rochpl PRIVATE roc::rocprim)

if(BUILD_TEST)
  target_link_libraries(rochpl PRIVATE GTest::GTest)
endif()


# Target properties
set_target_properties(rochpl PROPERTIES VERSION ${rochpl_VERSION})
if(BUILD_TEST)
  set_target_properties(rochpl PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  set_target_properties(rochpl PROPERTIES OUTPUT_NAME "rochpl-test")
else()
  set_target_properties(rochpl PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
set_target_properties(rochpl PROPERTIES DEBUG_POSTFIX "-d")

# Install targets
rocm_install_targets(TARGETS rochpl
                     PREFIX rochpl)

# Symbolic links
rocm_install_symlink_subdir(rochpl)

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-dev (>= 3.5.0)")
set(CPACK_RPM_PACKAGE_REQUIRES "rocm-dev >= 3.5.0")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name rochpl)

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute HPL application"
  MAINTAINER "Noel Chalmers")
