# ==============================================================================
# MLU (Cambricon MLU) Backend Configuration
# ==============================================================================

message(STATUS "Configuring MLU backend...")

if(NOT DEFINED TorchMLU_ROOT)
  execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import torch_mlu;print(torch_mlu.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TorchMLU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

include(FindPackageHandleStandardArgs)

find_package(TorchMLU CONFIG REQUIRED)

if(TorchMLU_FOUND)
  message(STATUS "Found MLU Runtime.")

  if(NOT TARGET MLU::mlu_runtime)
    add_library(MLU::mlu_runtime INTERFACE IMPORTED)
    set_target_properties(MLU::mlu_runtime PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MLU_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${TORCH_MLU_LIBRARIES}"
    )
    message(STATUS "Created MLU::mlu_runtime imported target")
  endif()
else()
  message(FATAL_ERROR "MLU Runtime not found. Please ensure cntoolkit is installed.")
endif()

function(target_link_mlu_libraries target_name)
  target_include_directories(${target_name} INTERFACE ${TORCH_MLU_INCLUDE_DIRS})
  target_link_libraries(${target_name} INTERFACE ${TORCH_MLU_LIBRARIES})
endfunction()

message(STATUS "MLU backend configuration complete")
