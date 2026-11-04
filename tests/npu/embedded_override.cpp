#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("expected scripts directory and kernel source");
    const std::string before = std::getenv("TRITON_ALL_BLOCKS_PARALLEL");
    const bool enabled = before != "1";
    triton_jit::ensure_initialized();
    py::gil_scoped_acquire gil;
    py::module_::import("sys").attr("path").attr("insert")(0, argv[1]);
    auto compiler = py::module_::import("standalone_compile");
    compiler.attr("torch").attr("npu").attr("set_device")(0);
    py::dict options;
    options["enable_auto_blockify"] = enabled;
    const auto directory =
        compiler.attr("compile_a_kernel")(argv[2], "counter_kernel", "*i32:16,121", 4, 1, 0, options)
            .cast<std::string>();
    const auto metadata = triton_jit::load_npu_metadata(directory, "counter_kernel");
    if (!metadata.enable_auto_blockify.has_value() || *metadata.enable_auto_blockify != enabled) {
      throw std::runtime_error("child compiler did not preserve the explicit option");
    }
    if (std::string(std::getenv("TRITON_ALL_BLOCKS_PARALLEL")) != before) {
      throw std::runtime_error("child compiler changed the caller's environment");
    }
    std::cout << "Embedded Python override compiled with mode " << enabled << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
