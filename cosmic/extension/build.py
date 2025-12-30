"""Build and load the CPU-only PyTorch C++ extension."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from torch.utils.cpp_extension import CppExtension, include_paths, load

_EXTENSION_NAME = "cosmic_cpu"


def extension_sources() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    csrc = root / "csrc"
    return [
        str(csrc / "cosmic_bindings.cpp"),
    ]


def _build_directory() -> str:
    path = Path(__file__).resolve().parent / "_build"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _make_cpp_extension(name: str) -> CppExtension:
    cxx_flags = [
        "-O3",
        "-std=c++17",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
        "-fno-trapping-math",
        "-fno-math-errno",
    ]
    return CppExtension(
        name=name,
        sources=extension_sources(),
        include_dirs=include_paths(),
        extra_compile_args={"cxx": cxx_flags},
    )


@lru_cache(maxsize=1)
def build_extension(verbose: bool = False) -> ModuleType:
    ext = _make_cpp_extension(_EXTENSION_NAME)
    return load(
        name=ext.name,
        sources=ext.sources,
        extra_cflags=ext.extra_compile_args["cxx"],
        extra_include_paths=ext.include_dirs,
        with_cuda=False,
        build_directory=_build_directory(),
        verbose=verbose,
    )


def build_extension_from_env() -> ModuleType:
    verbose = os.environ.get("COSMIC_BUILD_VERBOSE", "0") == "1"
    return build_extension(verbose=verbose)
