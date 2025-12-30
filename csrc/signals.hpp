#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>

namespace cosmic {

inline double tier_signal(const torch::Tensor& grad) {
    TORCH_CHECK(!grad.is_cuda(), "tier_signal: CPU-only; CUDA tensors are not supported");
    TORCH_CHECK(grad.is_floating_point(), "tier_signal: expected floating-point tensor");
    auto contiguous = grad.contiguous();
    const auto numel = contiguous.numel();
    if (numel == 0) {
        return 0.0;
    }

    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;

    AT_DISPATCH_FLOATING_TYPES(contiguous.scalar_type(), "tier_signal", [&] {
        const auto* ptr = contiguous.data_ptr<scalar_t>();
        for (int64_t i = 0; i < numel; ++i) {
            const double v = static_cast<double>(ptr[i]);
            const double av = std::abs(v);
            sum += v;
            sum_abs += av;
            sum_sq += v * v;
            if (av > max_abs) {
                max_abs = av;
            }
        }
    });

    const double inv = 1.0 / static_cast<double>(numel);
    const double mean = sum * inv;
    const double mean_abs = sum_abs * inv;
    const double mean_sq = sum_sq * inv;
    const double var = std::max(0.0, mean_sq - mean * mean);
    const double eps = 1e-12;
    const double var_ratio = var / (mean_abs * mean_abs + eps);
    const double max_ratio = max_abs / (mean_abs + eps);
    const double rms = std::sqrt(mean_sq);

    double signal = mean_abs + 0.1 * rms;
    const double var_boost = 1.0 + 0.25 * std::min(var_ratio, 8.0);
    const double max_boost = 1.0 + 0.1 * std::min(max_ratio, 8.0);
    signal *= var_boost * max_boost;
    return signal;
}

}  // namespace cosmic
