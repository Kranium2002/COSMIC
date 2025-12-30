#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(__has_include)
#if __has_include(<ATen/Parallel.h>)
#include <ATen/Parallel.h>
#define COSMIC_HAS_AT_PARALLEL 1
#elif __has_include(<c10/util/Parallel.h>)
#include <c10/util/Parallel.h>
#define COSMIC_HAS_C10_PARALLEL 1
#endif
#endif

namespace cosmic {

template <typename Func>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const Func& func) {
#if defined(COSMIC_HAS_AT_PARALLEL)
    at::parallel_for(begin, end, grain_size, func);
#elif defined(COSMIC_HAS_C10_PARALLEL)
    c10::parallel_for(begin, end, grain_size, func);
#else
    func(begin, end);
#endif
}

inline int8_t quant4_decode(uint8_t byte, bool high) {
    const uint8_t nibble = high ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                                : static_cast<uint8_t>(byte & 0x0F);
    return static_cast<int8_t>(static_cast<int>(nibble) - 8);
}

inline uint8_t quant4_encode(double value, double scale) {
    if (scale <= 0.0) {
        return static_cast<uint8_t>(8);
    }
    const int q = static_cast<int>(std::lrint(value / scale));
    const int q_clamped = std::max(-7, std::min(7, q));
    return static_cast<uint8_t>(q_clamped + 8);
}

/**
 * Fused COSMIC dual-EMA step kernel.
 * 
 * This is the core of COSMIC's novel algorithm, fused into a single cache-efficient pass:
 * 
 * Algorithm (per-element):
 *   1. ema_short = decay_short * ema_short + (1 - decay_short) * grad
 *   2. ema_long  = decay_long  * ema_long  + (1 - decay_long)  * grad
 *   3. exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
 *   4. momentum = blend * ema_short + (1 - blend) * ema_long
 *   5. denom = sqrt(exp_avg_sq / bias_correction2) + eps
 *   6. param = param - lr * momentum / denom - lr * wd * param
 * 
 * The dual-EMA (steps 1-2) is what makes COSMIC unique - it blends short-horizon
 * and long-horizon momentum based on parameter tier.
 */
template <typename scalar_t>
inline void cosmic_dual_ema_step_kernel(
    scalar_t* __restrict__ param_ptr,
    const scalar_t* __restrict__ grad_ptr,
    scalar_t* __restrict__ ema_short_ptr,
    scalar_t* __restrict__ ema_long_ptr,
    scalar_t* __restrict__ exp_avg_sq_ptr,
    int64_t begin,
    int64_t end,
    scalar_t lr,
    scalar_t decay_short,
    scalar_t decay_long,
    scalar_t beta2,
    scalar_t blend,           // How much to use short vs long EMA (0=long, 1=short)
    scalar_t bias_correction2,
    scalar_t eps,
    scalar_t weight_decay) {
    
    const scalar_t alpha_short = static_cast<scalar_t>(1) - decay_short;
    const scalar_t alpha_long = static_cast<scalar_t>(1) - decay_long;
    const scalar_t one_minus_beta2 = static_cast<scalar_t>(1) - beta2;
    const scalar_t bias_correction2_sqrt = std::sqrt(bias_correction2);
    
    for (int64_t i = begin; i < end; ++i) {
        const scalar_t grad = grad_ptr[i];
        
        // Update dual EMAs (COSMIC's novel momentum)
        scalar_t ema_s = ema_short_ptr[i];
        scalar_t ema_l = ema_long_ptr[i];
        ema_s = decay_short * ema_s + alpha_short * grad;
        ema_l = decay_long * ema_l + alpha_long * grad;
        ema_short_ptr[i] = ema_s;
        ema_long_ptr[i] = ema_l;
        
        // Blend short and long EMA based on tier
        scalar_t momentum;
        if (blend == static_cast<scalar_t>(1)) {
            momentum = ema_s;
        } else if (blend == static_cast<scalar_t>(0)) {
            momentum = ema_l;
        } else {
            momentum = blend * ema_s + (static_cast<scalar_t>(1) - blend) * ema_l;
        }
        
        // Update second moment (adaptive learning rate like Adam)
        scalar_t exp_sq = exp_avg_sq_ptr[i];
        exp_sq = beta2 * exp_sq + one_minus_beta2 * grad * grad;
        exp_avg_sq_ptr[i] = exp_sq;
        
        // Compute adaptive denominator with bias correction
        const scalar_t denom = std::sqrt(exp_sq) / bias_correction2_sqrt + eps;
        
        // Update parameter with decoupled weight decay (AdamW style)
        scalar_t p = param_ptr[i];
        if (weight_decay != static_cast<scalar_t>(0)) {
            p = p - lr * weight_decay * p;
        }
        p = p - lr * momentum / denom;
        param_ptr[i] = p;
    }
}

/**
 * Fused multi-tensor COSMIC step.
 * 
 * Processes all parameters in a single call, minimizing Python overhead.
 * Each parameter can have a different tier blend value and effective LR.
 * 
 * Args:
 *   params: Parameter tensors
 *   grads: Gradient tensors
 *   ema_shorts: Short-horizon EMA tensors
 *   ema_longs: Long-horizon EMA tensors  
 *   exp_avg_sqs: Second moment tensors
 *   blends: Per-parameter blend values (0=use long EMA, 1=use short EMA)
 *   effective_lrs: Per-parameter effective learning rates (after tier/gate scaling)
 *   step: Current optimization step (1-indexed)
 *   decay_short: Short EMA decay (typically 0.9)
 *   decay_long: Long EMA decay (typically 0.99)
 *   beta2: Second moment decay (typically 0.999)
 *   eps: Numerical stability term
 *   weight_decay: Decoupled weight decay
 */
inline void cosmic_fused_step(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_shorts,
    std::vector<torch::Tensor>& ema_longs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    const std::vector<double>& blends,
    const std::vector<double>& effective_lrs,
    int64_t step,
    double decay_short,
    double decay_long,
    double beta2,
    double eps,
    double weight_decay) {
    
    const size_t n = params.size();
    TORCH_CHECK(grads.size() == n, "cosmic_fused_step: grads must match params length");
    TORCH_CHECK(ema_shorts.size() == n, "cosmic_fused_step: ema_shorts must match params length");
    TORCH_CHECK(ema_longs.size() == n, "cosmic_fused_step: ema_longs must match params length");
    TORCH_CHECK(exp_avg_sqs.size() == n, "cosmic_fused_step: exp_avg_sqs must match params length");
    TORCH_CHECK(blends.size() == n, "cosmic_fused_step: blends must match params length");
    TORCH_CHECK(effective_lrs.size() == n, "cosmic_fused_step: effective_lrs must match params length");
    TORCH_CHECK(step >= 1, "cosmic_fused_step: step must be >= 1");
    
    // Compute bias correction for second moment
    const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    
    constexpr int64_t kParallelThreshold = 32768;
    constexpr int64_t kGrainSize = 4096;
    
    for (size_t i = 0; i < n; ++i) {
        auto& param = params[i];
        const auto& grad = grads[i];
        auto& ema_short = ema_shorts[i];
        auto& ema_long = ema_longs[i];
        auto& exp_avg_sq = exp_avg_sqs[i];
        const double blend = blends[i];
        const double lr = effective_lrs[i];
        
        // Skip if effective LR is zero (gated out)
        if (lr <= 0.0) {
            continue;
        }
        
        TORCH_CHECK(param.is_contiguous(), "cosmic_fused_step: param must be contiguous");
        TORCH_CHECK(grad.is_contiguous(), "cosmic_fused_step: grad must be contiguous");
        TORCH_CHECK(!param.is_cuda(), "cosmic_fused_step: CPU-only");
        
        const int64_t numel = param.numel();
        
        // Initialize state tensors if needed
        if (!ema_short.defined() || ema_short.numel() != numel) {
            ema_short = grad.clone();  // Initialize to first gradient
            ema_shorts[i] = ema_short;
        }
        if (!ema_long.defined() || ema_long.numel() != numel) {
            ema_long = grad.clone();   // Initialize to first gradient
            ema_longs[i] = ema_long;
        }
        if (!exp_avg_sq.defined() || exp_avg_sq.numel() != numel) {
            exp_avg_sq = torch::zeros_like(param);
            exp_avg_sqs[i] = exp_avg_sq;
        }
        
        AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "cosmic_fused_step", [&] {
            auto* param_ptr = param.data_ptr<scalar_t>();
            const auto* grad_ptr = grad.data_ptr<scalar_t>();
            auto* ema_short_ptr = ema_short.data_ptr<scalar_t>();
            auto* ema_long_ptr = ema_long.data_ptr<scalar_t>();
            auto* exp_avg_sq_ptr = exp_avg_sq.data_ptr<scalar_t>();
            
            const scalar_t lr_t = static_cast<scalar_t>(lr);
            const scalar_t decay_short_t = static_cast<scalar_t>(decay_short);
            const scalar_t decay_long_t = static_cast<scalar_t>(decay_long);
            const scalar_t beta2_t = static_cast<scalar_t>(beta2);
            const scalar_t blend_t = static_cast<scalar_t>(blend);
            const scalar_t bc2_t = static_cast<scalar_t>(bias_correction2);
            const scalar_t eps_t = static_cast<scalar_t>(eps);
            const scalar_t wd_t = static_cast<scalar_t>(weight_decay);
            
            if (numel >= kParallelThreshold) {
                parallel_for(0, numel, kGrainSize, [&](int64_t begin, int64_t end) {
                    cosmic_dual_ema_step_kernel(
                        param_ptr, grad_ptr, ema_short_ptr, ema_long_ptr, exp_avg_sq_ptr,
                        begin, end, lr_t, decay_short_t, decay_long_t, beta2_t,
                        blend_t, bc2_t, eps_t, wd_t);
                });
            } else {
                cosmic_dual_ema_step_kernel(
                    param_ptr, grad_ptr, ema_short_ptr, ema_long_ptr, exp_avg_sq_ptr,
                    0, numel, lr_t, decay_short_t, decay_long_t, beta2_t,
                    blend_t, bc2_t, eps_t, wd_t);
            }
        });
    }
}

template <typename scalar_t>
inline void cosmic_sign_step_kernel(
    scalar_t* __restrict__ param_ptr,
    const scalar_t* __restrict__ grad_ptr,
    int64_t begin,
    int64_t end,
    scalar_t lr,
    scalar_t weight_decay) {
    for (int64_t i = begin; i < end; ++i) {
        const scalar_t grad = grad_ptr[i];
        const scalar_t sign = grad > static_cast<scalar_t>(0)
            ? static_cast<scalar_t>(1)
            : (grad < static_cast<scalar_t>(0) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(0));

        scalar_t p = param_ptr[i];
        if (weight_decay != static_cast<scalar_t>(0)) {
            p = p - lr * weight_decay * p;
        }
        p = p - lr * sign;
        param_ptr[i] = p;
    }
}

inline void cosmic_fused_step_sign(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const std::vector<double>& effective_lrs,
    double weight_decay) {
    const size_t n = params.size();
    TORCH_CHECK(grads.size() == n, "cosmic_fused_step_sign: grads must match params length");
    TORCH_CHECK(effective_lrs.size() == n, "cosmic_fused_step_sign: effective_lrs must match params length");

    constexpr int64_t kParallelThreshold = 32768;
    constexpr int64_t kGrainSize = 4096;

    for (size_t i = 0; i < n; ++i) {
        auto& param = params[i];
        const auto& grad = grads[i];
        const double lr = effective_lrs[i];

        if (lr <= 0.0) {
            continue;
        }

        TORCH_CHECK(param.is_contiguous(), "cosmic_fused_step_sign: param must be contiguous");
        TORCH_CHECK(grad.is_contiguous(), "cosmic_fused_step_sign: grad must be contiguous");
        TORCH_CHECK(!param.is_cuda(), "cosmic_fused_step_sign: CPU-only");

        const int64_t numel = param.numel();

        AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "cosmic_fused_step_sign", [&] {
            auto* param_ptr = param.data_ptr<scalar_t>();
            const auto* grad_ptr = grad.data_ptr<scalar_t>();

            const scalar_t lr_t = static_cast<scalar_t>(lr);
            const scalar_t wd_t = static_cast<scalar_t>(weight_decay);

            if (numel >= kParallelThreshold) {
                parallel_for(0, numel, kGrainSize, [&](int64_t begin, int64_t end) {
                    cosmic_sign_step_kernel(
                        param_ptr, grad_ptr, begin, end, lr_t, wd_t);
                });
            } else {
                cosmic_sign_step_kernel(
                    param_ptr, grad_ptr, 0, numel, lr_t, wd_t);
            }
        });
    }
}

/**
 * Fused COSMIC step with quantized EMA state (4-bit packed, block-wise scales).
 *
 * This kernel keeps EMA short/long in 4-bit packed format with per-block scales,
 * while performing updates in floating point. It updates params, EMA state, and
 * second-moment state in a single pass per block.
 */
inline void cosmic_fused_step_quant4(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_short_qs,
    std::vector<torch::Tensor>& ema_short_scales,
    std::vector<torch::Tensor>& ema_long_qs,
    std::vector<torch::Tensor>& ema_long_scales,
    std::vector<torch::Tensor>& exp_avg_sqs,
    const std::vector<double>& blends,
    const std::vector<double>& effective_lrs,
    int64_t step,
    double decay_short,
    double decay_long,
    double beta2,
    double eps,
    double weight_decay,
    int64_t block_size) {
    const size_t n = params.size();
    TORCH_CHECK(grads.size() == n, "cosmic_fused_step_quant4: grads must match params length");
    TORCH_CHECK(ema_short_qs.size() == n, "cosmic_fused_step_quant4: ema_short_qs must match params length");
    TORCH_CHECK(ema_short_scales.size() == n, "cosmic_fused_step_quant4: ema_short_scales must match params length");
    TORCH_CHECK(ema_long_qs.size() == n, "cosmic_fused_step_quant4: ema_long_qs must match params length");
    TORCH_CHECK(ema_long_scales.size() == n, "cosmic_fused_step_quant4: ema_long_scales must match params length");
    TORCH_CHECK(exp_avg_sqs.size() == n, "cosmic_fused_step_quant4: exp_avg_sqs must match params length");
    TORCH_CHECK(blends.size() == n, "cosmic_fused_step_quant4: blends must match params length");
    TORCH_CHECK(effective_lrs.size() == n, "cosmic_fused_step_quant4: effective_lrs must match params length");
    TORCH_CHECK(step >= 1, "cosmic_fused_step_quant4: step must be >= 1");
    TORCH_CHECK(block_size > 0, "cosmic_fused_step_quant4: block_size must be > 0");
    TORCH_CHECK(block_size % 2 == 0, "cosmic_fused_step_quant4: block_size must be even");

    const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    const double bias_correction2_sqrt = std::sqrt(bias_correction2);
    const double alpha_short = 1.0 - decay_short;
    const double alpha_long = 1.0 - decay_long;
    const double one_minus_beta2 = 1.0 - beta2;

    for (size_t i = 0; i < n; ++i) {
        auto& param = params[i];
        const auto& grad = grads[i];
        auto& ema_short_q = ema_short_qs[i];
        auto& ema_short_scale = ema_short_scales[i];
        auto& ema_long_q = ema_long_qs[i];
        auto& ema_long_scale = ema_long_scales[i];
        auto& exp_avg_sq = exp_avg_sqs[i];
        const double blend = blends[i];
        const double lr = effective_lrs[i];

        if (lr <= 0.0) {
            continue;
        }

        TORCH_CHECK(param.is_contiguous(), "cosmic_fused_step_quant4: param must be contiguous");
        TORCH_CHECK(grad.is_contiguous(), "cosmic_fused_step_quant4: grad must be contiguous");
        TORCH_CHECK(!param.is_cuda(), "cosmic_fused_step_quant4: CPU-only");
        TORCH_CHECK(ema_short_q.is_contiguous(), "cosmic_fused_step_quant4: ema_short_q must be contiguous");
        TORCH_CHECK(ema_long_q.is_contiguous(), "cosmic_fused_step_quant4: ema_long_q must be contiguous");
        TORCH_CHECK(ema_short_scale.is_contiguous(), "cosmic_fused_step_quant4: ema_short_scale must be contiguous");
        TORCH_CHECK(ema_long_scale.is_contiguous(), "cosmic_fused_step_quant4: ema_long_scale must be contiguous");
        TORCH_CHECK(ema_short_q.scalar_type() == torch::kUInt8, "cosmic_fused_step_quant4: ema_short_q must be uint8");
        TORCH_CHECK(ema_long_q.scalar_type() == torch::kUInt8, "cosmic_fused_step_quant4: ema_long_q must be uint8");
        TORCH_CHECK(ema_short_scale.scalar_type() == torch::kFloat32, "cosmic_fused_step_quant4: ema_short_scale must be float32");
        TORCH_CHECK(ema_long_scale.scalar_type() == torch::kFloat32, "cosmic_fused_step_quant4: ema_long_scale must be float32");

        const int64_t numel = param.numel();
        const int64_t num_blocks = (numel + block_size - 1) / block_size;
        TORCH_CHECK(ema_short_scale.numel() == num_blocks, "cosmic_fused_step_quant4: ema_short_scale size mismatch");
        TORCH_CHECK(ema_long_scale.numel() == num_blocks, "cosmic_fused_step_quant4: ema_long_scale size mismatch");
        TORCH_CHECK(ema_short_q.numel() == (numel + 1) / 2, "cosmic_fused_step_quant4: ema_short_q size mismatch");
        TORCH_CHECK(ema_long_q.numel() == (numel + 1) / 2, "cosmic_fused_step_quant4: ema_long_q size mismatch");
        TORCH_CHECK(exp_avg_sq.numel() == numel, "cosmic_fused_step_quant4: exp_avg_sq size mismatch");

        AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "cosmic_fused_step_quant4", [&] {
            auto* param_ptr = param.data_ptr<scalar_t>();
            const auto* grad_ptr = grad.data_ptr<scalar_t>();
            auto* exp_avg_sq_ptr = exp_avg_sq.data_ptr<scalar_t>();
            auto* ema_short_q_ptr = ema_short_q.data_ptr<uint8_t>();
            auto* ema_long_q_ptr = ema_long_q.data_ptr<uint8_t>();
            auto* ema_short_scale_ptr = ema_short_scale.data_ptr<float>();
            auto* ema_long_scale_ptr = ema_long_scale.data_ptr<float>();

            std::vector<scalar_t> ema_short_buf(block_size);
            std::vector<scalar_t> ema_long_buf(block_size);

            const scalar_t lr_t = static_cast<scalar_t>(lr);
            const scalar_t decay_short_t = static_cast<scalar_t>(decay_short);
            const scalar_t decay_long_t = static_cast<scalar_t>(decay_long);
            const scalar_t alpha_short_t = static_cast<scalar_t>(alpha_short);
            const scalar_t alpha_long_t = static_cast<scalar_t>(alpha_long);
            const scalar_t beta2_t = static_cast<scalar_t>(beta2);
            const scalar_t one_minus_beta2_t = static_cast<scalar_t>(one_minus_beta2);
            const scalar_t blend_t = static_cast<scalar_t>(blend);
            const scalar_t eps_t = static_cast<scalar_t>(eps);
            const scalar_t wd_t = static_cast<scalar_t>(weight_decay);
            const scalar_t bc2_sqrt_t = static_cast<scalar_t>(bias_correction2_sqrt);

            for (int64_t block = 0; block < num_blocks; ++block) {
                const int64_t start = block * block_size;
                const int64_t end = std::min(start + block_size, numel);
                const int64_t block_len = end - start;

                const float scale_short = ema_short_scale_ptr[block];
                const float scale_long = ema_long_scale_ptr[block];
                const bool init_short = scale_short <= 0.0f;
                const bool init_long = scale_long <= 0.0f;

                scalar_t max_abs_short = static_cast<scalar_t>(0);
                scalar_t max_abs_long = static_cast<scalar_t>(0);

                for (int64_t offset = 0; offset < block_len; ++offset) {
                    const int64_t idx = start + offset;
                    const scalar_t grad_val = grad_ptr[idx];

                    scalar_t ema_s = grad_val;
                    scalar_t ema_l = grad_val;
                    if (!init_short) {
                        const uint8_t byte = ema_short_q_ptr[idx / 2];
                        const int8_t q = quant4_decode(byte, (idx & 1) != 0);
                        ema_s = static_cast<scalar_t>(static_cast<float>(q) * scale_short);
                    }
                    if (!init_long) {
                        const uint8_t byte = ema_long_q_ptr[idx / 2];
                        const int8_t q = quant4_decode(byte, (idx & 1) != 0);
                        ema_l = static_cast<scalar_t>(static_cast<float>(q) * scale_long);
                    }

                    ema_s = decay_short_t * ema_s + alpha_short_t * grad_val;
                    ema_l = decay_long_t * ema_l + alpha_long_t * grad_val;

                    scalar_t exp_sq = exp_avg_sq_ptr[idx];
                    exp_sq = beta2_t * exp_sq + one_minus_beta2_t * grad_val * grad_val;
                    exp_avg_sq_ptr[idx] = exp_sq;

                    scalar_t momentum;
                    if (blend_t == static_cast<scalar_t>(1)) {
                        momentum = ema_s;
                    } else if (blend_t == static_cast<scalar_t>(0)) {
                        momentum = ema_l;
                    } else {
                        momentum = blend_t * ema_s + (static_cast<scalar_t>(1) - blend_t) * ema_l;
                    }

                    const scalar_t denom = std::sqrt(exp_sq) / bc2_sqrt_t + eps_t;
                    scalar_t p = param_ptr[idx];
                    if (wd_t != static_cast<scalar_t>(0)) {
                        p = p - lr_t * wd_t * p;
                    }
                    p = p - lr_t * momentum / denom;
                    param_ptr[idx] = p;

                    ema_short_buf[offset] = ema_s;
                    ema_long_buf[offset] = ema_l;

                    const scalar_t abs_s = std::abs(ema_s);
                    const scalar_t abs_l = std::abs(ema_l);
                    if (abs_s > max_abs_short) {
                        max_abs_short = abs_s;
                    }
                    if (abs_l > max_abs_long) {
                        max_abs_long = abs_l;
                    }
                }

                const double new_scale_short = max_abs_short > static_cast<scalar_t>(0)
                    ? static_cast<double>(max_abs_short) / 7.0
                    : 1.0;
                const double new_scale_long = max_abs_long > static_cast<scalar_t>(0)
                    ? static_cast<double>(max_abs_long) / 7.0
                    : 1.0;

                ema_short_scale_ptr[block] = static_cast<float>(new_scale_short);
                ema_long_scale_ptr[block] = static_cast<float>(new_scale_long);

                for (int64_t offset = 0; offset < block_len; offset += 2) {
                    const int64_t idx = start + offset;
                    const uint8_t q0_s = quant4_encode(static_cast<double>(ema_short_buf[offset]), new_scale_short);
                    uint8_t q1_s = static_cast<uint8_t>(8);
                    if (offset + 1 < block_len) {
                        q1_s = quant4_encode(static_cast<double>(ema_short_buf[offset + 1]), new_scale_short);
                    }
                    ema_short_q_ptr[idx / 2] = static_cast<uint8_t>((q1_s << 4) | (q0_s & 0x0F));

                    const uint8_t q0_l = quant4_encode(static_cast<double>(ema_long_buf[offset]), new_scale_long);
                    uint8_t q1_l = static_cast<uint8_t>(8);
                    if (offset + 1 < block_len) {
                        q1_l = quant4_encode(static_cast<double>(ema_long_buf[offset + 1]), new_scale_long);
                    }
                    ema_long_q_ptr[idx / 2] = static_cast<uint8_t>((q1_l << 4) | (q0_l & 0x0F));
                }
            }
        });
    }
}

}  // namespace cosmic
