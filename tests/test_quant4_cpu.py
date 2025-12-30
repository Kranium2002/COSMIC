import torch

from cosmic.extension import (
    quant4_pack,
    quant4_pack_blocks,
    quant4_unpack,
    quant4_unpack_blocks,
)


def test_quant4_roundtrip():
    torch.manual_seed(0)
    tensor = torch.randn(17, dtype=torch.float32)
    packed, scale, numel = quant4_pack(tensor)
    unpacked = quant4_unpack(packed, scale, numel)

    assert unpacked.shape == tensor.shape
    max_error = (tensor - unpacked).abs().max().item()
    assert max_error <= scale + 1e-6


def test_quant4_blocks_roundtrip():
    torch.manual_seed(0)
    tensor = torch.randn(129, dtype=torch.float32)
    packed, scales, numel, block_size, use_zero_point = quant4_pack_blocks(tensor, 32)
    unpacked = quant4_unpack_blocks(packed, scales, numel, block_size, use_zero_point)

    assert unpacked.shape == tensor.shape
    max_error = (tensor - unpacked).abs().max().item()
    assert max_error <= scales.max().item() + 1e-6
