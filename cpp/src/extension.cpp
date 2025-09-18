#include <torch/extension.h>

#include "torpedocode/kernels.hpp"

namespace torpedocode {

#ifndef TORPEDOCODE_ENABLE_CUDA
// Provide a CPU fallback stub when building without CUDA to satisfy linker
HybridForwardOutputs hybrid_forward_cuda(const HybridForwardInputs &inputs) {
  return hybrid_forward_cpu(inputs);
}
#endif

HybridForwardOutputs hybrid_forward_cpu(const HybridForwardInputs &inputs) {
  auto features = inputs.features.contiguous();
  auto topology = inputs.topology;
  if (features.sizes() != topology.sizes()) {
    if (topology.dim() >= 1) {
      auto reduced = topology.mean(-1, /*keepdim=*/true);
      topology = reduced.expand_as(features).contiguous();
    }
  } else {
    topology = topology.contiguous();
  }
  auto fused = features + topology;
  if (inputs.weights.defined() && inputs.weights.numel() == fused.numel()) {
    fused = fused * inputs.weights.contiguous().view_as(fused);
  }
  return HybridForwardOutputs{fused};
}

HybridForwardOutputs hybrid_forward(const HybridForwardInputs &inputs) {
  if (inputs.features.is_cuda()) {
    return hybrid_forward_cuda(inputs);
  }
  return hybrid_forward_cpu(inputs);
}

} // namespace torpedocode

// Wrapper with Tensor signature for Torch op registration.
// Python tests index [0], so return a tuple with the fused tensor at index 0.
// Schema exposes two tensors to avoid single-return unwrapping in some runtimes.
static std::tuple<torch::Tensor, torch::Tensor> hybrid_forward_tensors(
    torch::Tensor features, torch::Tensor topology, torch::Tensor weights) {
  torpedocode::HybridForwardInputs in{features, topology, weights};
  auto fused = torpedocode::hybrid_forward(in).fused;
  return std::make_tuple(fused, fused);
}

TORCH_LIBRARY(torpedocode, m) {
  m.def(
      "hybrid_forward(Tensor features, Tensor topology, Tensor weights) -> (Tensor, Tensor)",
        &hybrid_forward_tensors);
}

// Define a minimal Python module so torch.utils.cpp_extension.load can import it.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Ops are registered via TORCH_LIBRARY above.
}
