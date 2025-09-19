#pragma once

#include <torch/extension.h>

namespace torpedocode {

struct HybridForwardInputs {
  torch::Tensor features;
  torch::Tensor topology;
  torch::Tensor weights;
};

struct HybridForwardOutputs {
  torch::Tensor fused;
};

HybridForwardOutputs hybrid_forward_cpu(const HybridForwardInputs &inputs);
HybridForwardOutputs hybrid_forward_cuda(const HybridForwardInputs &inputs);
HybridForwardOutputs hybrid_forward(const HybridForwardInputs &inputs);

enum class SmoothnessNorm : int64_t {
  None = 0,
  Global = 1,
  PerSequence = 2,
};

struct TPPLossInputs {
  torch::Tensor intensities;
  torch::Tensor mark_mu;
  torch::Tensor mark_log_sigma;
  torch::Tensor event_types;
  torch::Tensor delta_t;
  torch::Tensor sizes;
  torch::Tensor mask;
  SmoothnessNorm smoothness_norm;
};

struct TPPLossOutputs {
  torch::Tensor nll_mean;
  torch::Tensor smoothness;
};

TPPLossOutputs tpp_loss(const TPPLossInputs &inputs);

} // namespace torpedocode
