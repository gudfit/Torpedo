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
  TORCH_CHECK(inputs.features.dtype() == torch::kFloat32,
              "features must be float32");
  TORCH_CHECK(inputs.topology.dtype() == torch::kFloat32,
              "topology must be float32");
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
    TORCH_CHECK(inputs.weights.dtype() == torch::kFloat32,
                "weights must be float32 when provided");
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
// Schema exposes two tensors to avoid single-return unwrapping in some
// runtimes.
static std::tuple<torch::Tensor, torch::Tensor>
hybrid_forward_tensors(torch::Tensor features, torch::Tensor topology,
                       torch::Tensor weights) {
  torpedocode::HybridForwardInputs in{features, topology, weights};
  auto fused = torpedocode::hybrid_forward(in).fused;
  return std::make_tuple(fused, fused);
}

TORCH_LIBRARY(torpedocode, m) {
  m.def("hybrid_forward(Tensor features, Tensor topology, Tensor weights) -> "
        "(Tensor, Tensor)",
        &hybrid_forward_tensors);
  // Alias for clarity
  m.def("fuse_add(Tensor features, Tensor topology, Tensor weights) -> "
        "(Tensor, Tensor)",
        &hybrid_forward_tensors);
  m.def("fuse_projected(Tensor features, Tensor topology, Tensor Wf, Tensor "
        "Wt, Tensor? bias) -> Tensor",
        [](torch::Tensor features, torch::Tensor topology, torch::Tensor Wf,
           torch::Tensor Wt, c10::optional<torch::Tensor> bias) {
          TORCH_CHECK(features.dim() == 2 && topology.dim() == 2,
                      "fuse_projected expects 2D tensors");
          TORCH_CHECK(features.size(0) == topology.size(0), "batch mismatch");
          // Keep device; convert dtype to f32 for compute
          auto opts_f32 = features.options().dtype(torch::kFloat32);
          auto f = features.to(opts_f32).contiguous();
          auto t = topology.to(opts_f32).contiguous();
          auto wf = Wf.to(opts_f32).contiguous();
          auto wt = Wt.to(opts_f32).contiguous();
          TORCH_CHECK(f.size(1) == wf.size(1) && t.size(1) == wt.size(1),
                      "projection dims mismatch");
          auto y = torch::matmul(f, wf.transpose(0, 1)) +
                   torch::matmul(t, wt.transpose(0, 1));
          if (bias.has_value() && bias->defined()) {
            auto b = bias->to(opts_f32).contiguous();
            if (b.dim() == 1) {
              y = y + b.unsqueeze(0);
            } else {
              y = y + b;
            }
          }
          // Return in original dtype on original device
          return y.to(features.dtype());
        });
  m.def("tpp_loss(Tensor intensities, Tensor mark_mu, Tensor mark_log_sigma, "
        "Tensor event_types, Tensor delta_t, Tensor? sizes=None, Tensor? "
        "mask=None, int smoothness_mode=1) -> (Tensor, Tensor)",
        [](torch::Tensor intensities, torch::Tensor mark_mu,
           torch::Tensor mark_log_sigma, torch::Tensor event_types,
           torch::Tensor delta_t, c10::optional<torch::Tensor> sizes,
           c10::optional<torch::Tensor> mask, int64_t smoothness_mode) {
          auto mode = static_cast<int64_t>(smoothness_mode);
          if (mode < 0 || mode > 2) {
            TORCH_WARN("Unknown smoothness_mode=", mode,
                       "; defaulting to global (1)");
            mode = 1;
          }
          torpedocode::TPPLossInputs in{intensities,
                                        mark_mu,
                                        mark_log_sigma,
                                        event_types,
                                        delta_t,
                                        sizes.has_value() ? *sizes : torch::Tensor(),
                                        mask.has_value() ? *mask : torch::Tensor(),
                                        static_cast<torpedocode::SmoothnessNorm>(mode)};
          auto out = torpedocode::tpp_loss(in);
          return std::make_tuple(out.nll_mean, out.smoothness);
        });
}

// Define a minimal Python module so torch.utils.cpp_extension.load can import
// it.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Ops are registered via TORCH_LIBRARY above.
}
