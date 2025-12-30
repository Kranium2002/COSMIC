#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace cosmic {

struct Quant4Metadata {
    torch::Tensor scales;
    torch::Tensor zero_points;
    int64_t numel;
    int64_t block_size;
    bool use_zero_point;
};

inline std::tuple<torch::Tensor, double, int64_t> quant4_pack(const torch::Tensor& input) {
    TORCH_CHECK(!input.is_cuda(), "quant4_pack: CPU-only; CUDA tensors are not supported");
    TORCH_CHECK(input.is_floating_point(), "quant4_pack: expected floating-point tensor");

    auto contiguous = input.contiguous();
    const int64_t numel = contiguous.numel();
    const double max_abs = contiguous.abs().max().item<double>();
    const double scale = max_abs > 0.0 ? max_abs / 7.0 : 1.0;

    auto packed = torch::zeros(
        {(numel + 1) / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    auto packed_ptr = packed.data_ptr<uint8_t>();

    AT_DISPATCH_FLOATING_TYPES(contiguous.scalar_type(), "quant4_pack", [&] {
        const auto* in_ptr = contiguous.data_ptr<scalar_t>();
        for (int64_t i = 0; i < numel; i += 2) {
            const double v0 = static_cast<double>(in_ptr[i]);
            const int q0 = static_cast<int>(std::lrint(v0 / scale));
            const int q0_clamped = std::max(-7, std::min(7, q0));
            uint8_t low = static_cast<uint8_t>(q0_clamped + 8);

            uint8_t high = 0;
            if (i + 1 < numel) {
                const double v1 = static_cast<double>(in_ptr[i + 1]);
                const int q1 = static_cast<int>(std::lrint(v1 / scale));
                const int q1_clamped = std::max(-7, std::min(7, q1));
                high = static_cast<uint8_t>(q1_clamped + 8);
            }

            packed_ptr[i / 2] = static_cast<uint8_t>((high << 4) | (low & 0x0F));
        }
    });

    return std::make_tuple(packed, scale, numel);
}

inline torch::Tensor quant4_unpack(
    const torch::Tensor& packed,
    double scale,
    int64_t numel) {
    TORCH_CHECK(!packed.is_cuda(), "quant4_unpack: CPU-only; CUDA tensors are not supported");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "quant4_unpack: expected uint8 tensor");

    auto output = torch::zeros({numel}, torch::TensorOptions().dtype(torch::kFloat32));
    auto out_ptr = output.data_ptr<float>();
    auto packed_ptr = packed.data_ptr<uint8_t>();

    for (int64_t i = 0; i < numel; i += 2) {
        const uint8_t byte = packed_ptr[i / 2];
        const int low = static_cast<int>(byte & 0x0F) - 8;
        const int high = static_cast<int>((byte >> 4) & 0x0F) - 8;

        out_ptr[i] = static_cast<float>(low * scale);
        if (i + 1 < numel) {
            out_ptr[i + 1] = static_cast<float>(high * scale);
        }
    }

    return output;
}

inline std::tuple<std::vector<torch::Tensor>, std::vector<double>, std::vector<int64_t>>
quant4_pack_many(const std::vector<torch::Tensor>& inputs) {
    std::vector<torch::Tensor> packed_list;
    std::vector<double> scales;
    std::vector<int64_t> numels;
    packed_list.reserve(inputs.size());
    scales.reserve(inputs.size());
    numels.reserve(inputs.size());

    for (const auto& input : inputs) {
        auto packed = torch::Tensor();
        double scale = 1.0;
        int64_t numel = 0;
        std::tie(packed, scale, numel) = quant4_pack(input);
        packed_list.push_back(packed);
        scales.push_back(scale);
        numels.push_back(numel);
    }

    return std::make_tuple(packed_list, scales, numels);
}

inline std::vector<torch::Tensor> quant4_unpack_many(
    const std::vector<torch::Tensor>& packed_list,
    const std::vector<double>& scales,
    const std::vector<int64_t>& numels) {
    TORCH_CHECK(
        packed_list.size() == scales.size() && packed_list.size() == numels.size(),
        "quant4_unpack_many: list sizes must match");
    std::vector<torch::Tensor> outputs;
    outputs.reserve(packed_list.size());
    for (size_t i = 0; i < packed_list.size(); ++i) {
        outputs.push_back(quant4_unpack(packed_list[i], scales[i], numels[i]));
    }
    return outputs;
}

inline std::tuple<torch::Tensor, Quant4Metadata> quant4_pack_blocks(
    const torch::Tensor& input,
    int64_t block_size,
    bool use_zero_point) {
    TORCH_CHECK(!input.is_cuda(), "quant4_pack_blocks: CPU-only; CUDA tensors are not supported");
    TORCH_CHECK(input.is_floating_point(), "quant4_pack_blocks: expected floating-point tensor");
    TORCH_CHECK(block_size > 0, "quant4_pack_blocks: block_size must be positive");

    auto contiguous = input.contiguous();
    const int64_t numel = contiguous.numel();
    const int64_t num_blocks = (numel + block_size - 1) / block_size;

    auto packed = torch::zeros(
        {(numel + 1) / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto scales = torch::zeros(
        {num_blocks},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto zero_points = torch::zeros(
        {num_blocks},
        torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));

    auto packed_ptr = packed.data_ptr<uint8_t>();
    auto scales_ptr = scales.data_ptr<float>();
    auto zero_ptr = zero_points.data_ptr<int16_t>();

    AT_DISPATCH_FLOATING_TYPES(contiguous.scalar_type(), "quant4_pack_blocks", [&] {
        const auto* in_ptr = contiguous.data_ptr<scalar_t>();
        for (int64_t block = 0; block < num_blocks; ++block) {
            const int64_t start = block * block_size;
            const int64_t end = std::min(start + block_size, numel);

            double min_val = 0.0;
            double max_val = 0.0;
            if (start < end) {
                min_val = static_cast<double>(in_ptr[start]);
                max_val = min_val;
                for (int64_t i = start + 1; i < end; ++i) {
                    const double v = static_cast<double>(in_ptr[i]);
                    min_val = std::min(min_val, v);
                    max_val = std::max(max_val, v);
                }
            }

            float scale = 1.0f;
            int16_t zero_point = 0;
            if (use_zero_point) {
                const double range = max_val - min_val;
                scale = range > 0.0 ? static_cast<float>(range / 15.0) : 1.0f;
                const double zp = scale > 0.0 ? -min_val / scale : 0.0;
                const int zp_rounded = static_cast<int>(std::lrint(zp));
                zero_point = static_cast<int16_t>(std::max(0, std::min(15, zp_rounded)));
            } else {
                const double max_abs = std::max(std::abs(min_val), std::abs(max_val));
                scale = max_abs > 0.0 ? static_cast<float>(max_abs / 7.0) : 1.0f;
                zero_point = 0;
            }

            scales_ptr[block] = scale;
            zero_ptr[block] = zero_point;

            for (int64_t i = start; i < end; i += 2) {
                const int64_t packed_index = i / 2;
                auto quantize = [&](double v) -> uint8_t {
                    if (use_zero_point) {
                        const double q = scale > 0.0 ? v / scale + zero_point : zero_point;
                        const int q_rounded = static_cast<int>(std::lrint(q));
                        const int q_clamped = std::max(0, std::min(15, q_rounded));
                        return static_cast<uint8_t>(q_clamped);
                    }
                    const double q = scale > 0.0 ? v / scale : 0.0;
                    const int q_rounded = static_cast<int>(std::lrint(q));
                    const int q_clamped = std::max(-7, std::min(7, q_rounded));
                    return static_cast<uint8_t>(q_clamped + 8);
                };

                const uint8_t low = quantize(static_cast<double>(in_ptr[i]));
                uint8_t high = 0;
                if (i + 1 < end) {
                    high = quantize(static_cast<double>(in_ptr[i + 1]));
                }
                packed_ptr[packed_index] = static_cast<uint8_t>((high << 4) | (low & 0x0F));
            }
        }
    });

    Quant4Metadata meta{scales, zero_points, numel, block_size, use_zero_point};
    return std::make_tuple(packed, meta);
}

inline torch::Tensor quant4_unpack_blocks(
    const torch::Tensor& packed,
    const Quant4Metadata& meta) {
    TORCH_CHECK(!packed.is_cuda(), "quant4_unpack_blocks: CPU-only; CUDA tensors are not supported");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "quant4_unpack_blocks: expected uint8");
    TORCH_CHECK(meta.scales.device().is_cpu(), "quant4_unpack_blocks: scales must be CPU");
    TORCH_CHECK(meta.zero_points.device().is_cpu(), "quant4_unpack_blocks: zero_points must be CPU");
    TORCH_CHECK(meta.block_size > 0, "quant4_unpack_blocks: block_size must be positive");

    auto output = torch::zeros(
        {meta.numel},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto out_ptr = output.data_ptr<float>();
    auto packed_ptr = packed.data_ptr<uint8_t>();
    auto scales_ptr = meta.scales.data_ptr<float>();
    auto zero_ptr = meta.zero_points.data_ptr<int16_t>();

    const int64_t num_blocks = (meta.numel + meta.block_size - 1) / meta.block_size;
    for (int64_t block = 0; block < num_blocks; ++block) {
        const int64_t start = block * meta.block_size;
        const int64_t end = std::min(start + meta.block_size, meta.numel);
        const float scale = scales_ptr[block];
        const int16_t zero_point = zero_ptr[block];

        for (int64_t i = start; i < end; i += 2) {
            const uint8_t byte = packed_ptr[i / 2];
            const uint8_t low = static_cast<uint8_t>(byte & 0x0F);
            const uint8_t high = static_cast<uint8_t>((byte >> 4) & 0x0F);

            if (meta.use_zero_point) {
                out_ptr[i] = static_cast<float>((static_cast<int>(low) - zero_point) * scale);
                if (i + 1 < end) {
                    out_ptr[i + 1] =
                        static_cast<float>((static_cast<int>(high) - zero_point) * scale);
                }
            } else {
                const int low_s = static_cast<int>(low) - 8;
                const int high_s = static_cast<int>(high) - 8;
                out_ptr[i] = static_cast<float>(low_s * scale);
                if (i + 1 < end) {
                    out_ptr[i + 1] = static_cast<float>(high_s * scale);
                }
            }
        }
    }

    return output;
}

}  // namespace cosmic
