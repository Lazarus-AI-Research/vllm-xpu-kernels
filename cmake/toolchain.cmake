# use this file to set the compiler and flags for SYCL

set(CMPLR_ROOT $ENV{CMPLR_ROOT})
if(NOT CMPLR_ROOT AND DEFINED ENV{SYCL_HOME})
  set(CMPLR_ROOT $ENV{SYCL_HOME})
endif()
message(STATUS "CMPLR_ROOT: ${CMPLR_ROOT}")

find_program(VLLM_XPU_ICPX icpx HINTS "${CMPLR_ROOT}/bin")
find_program(VLLM_XPU_ICX icx HINTS "${CMPLR_ROOT}/bin")
if(NOT VLLM_XPU_ICPX OR NOT VLLM_XPU_ICX)
  message(
    FATAL_ERROR
      "Could not find Intel oneAPI icx/icpx compilers. Install the DPC++ "
      "compiler and/or set CMPLR_ROOT to its oneAPI compiler root.")
endif()

set(CMAKE_CXX_COMPILER ${VLLM_XPU_ICPX})
set(CMAKE_C_COMPILER ${VLLM_XPU_ICX})
