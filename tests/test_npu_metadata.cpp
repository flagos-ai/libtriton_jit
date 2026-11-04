#include "triton_jit/kernel_metadata.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected fixture directory");
    const std::string directory = argv[1];
    for (const auto* name : {"enabled", "disabled"}) {
      const auto metadata = triton_jit::load_npu_metadata(directory, name);
      const bool expected = std::string(name) == "enabled";
      if (!metadata.enable_auto_blockify.has_value() || *metadata.enable_auto_blockify != expected ||
          metadata.mix_mode != "aiv") {
        throw std::runtime_error(std::string("incorrect compiled mode: ") + name);
      }
    }
    for (const auto* name : {"null", "missing", "invalid", "absent_file"}) {
      if (triton_jit::load_npu_metadata(directory, name).enable_auto_blockify.has_value()) {
        throw std::runtime_error(std::string("ambiguous metadata guessed a mode: ") + name);
      }
    }
    std::cout << "NPU compiled mode metadata: all checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
