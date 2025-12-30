# COSMIC

COSMIC (Capital-Optimized State & Momentum with Intelligent Calibration) is a CPU-only PyTorch optimizer designed for research and future production use. It ships a v1 C++ extension and a clean Python API that you can run and test on a CPU-only machine.

COSMIC treats optimizer state like a capital budget: full-history state is reserved for parameters that benefit most, while lighter-weight updates are used elsewhere.

## Features (v1)
- CPU-only PyTorch C++ extension (no CUDA support)
- Multi-tensor update entrypoint for deterministic, in-place parameter updates
- Tiered state allocation (Tier-0 / Tier-1 / Tier-2) for memory-efficient training
- Gating that downscales updates when gradient signals are noisy or unstable
- Optional 4-bit optimizer-state quantization helpers
- Structured logging helpers for reproducible experiments
- LLM finetuning benchmark script

## Technique (short)
COSMIC allocates optimizer "capital" across tiers:
- Tier 2: full history (dual EMA + adaptive denom) for high-signal params
- Tier 1: scalar EMA magnitude + sign updates (tiny state, retains scale info)
- Tier 0: sign-only updates (no momentum state)

Gating uses recent gradient statistics to scale learning rates when signals are volatile, helping stability without spending extra state.

## Coming soon
A detailed technical blog on how COSMIC works and GPU support are coming soon.

## Requirements
- Python 3.12
- PyTorch 2.1+ (CPU)
- Poetry

## Install

```bash
poetry install
```

Or, for end users who just want to use the package:

```bash
pip install cosmic
```

No manual build step is required. The C++ extension compiles automatically on first use.
Note: COSMIC does not ship prebuilt wheels yet, so users will build the extension from source locally.

## Runtime build prerequisites (CPU-only)
The C++ extension is compiled automatically on first use. Make sure you have:
- A C++17 compiler toolchain
- `ninja` available in your environment
- Python development headers for your interpreter

On Debian/Ubuntu, a minimal setup looks like:

```bash
sudo apt-get install -y build-essential python3.12-dev
```

Note: On some systems you may need to install the CPU-only PyTorch wheel explicitly, for example:

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Build the extension

```bash
poetry run python -c "from cosmic.extension import get_extension; get_extension(verbose=True)"
```

## Quick start

```python
import torch
from cosmic import Cosmic

torch.manual_seed(0)
model = torch.nn.Linear(4, 2)
optimizer = Cosmic(model.parameters(), lr=1e-2)

x = torch.randn(8, 4)
y = torch.randn(8, 2)

for _ in range(5):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
```

## Tests

```bash
poetry run pytest
```

## Benchmark

The benchmark script runs one optimizer per run; edit the `optimizers` list in
`benchmark_llm_finetune.py` to compare COSMIC vs Adam.

```bash
poetry run python benchmark_llm_finetune.py
```

Recent run (CPU-only, 12 threads, single process per optimizer):

```
Model: distilgpt2 (81,912,576 params)
Dataset: wikitext-2-raw-v1 (train[:1%])
Steps: 600, batch=4, block=128, lr=5e-4, weight_decay=0.01

optimizer    loss    ppl   time(s)   tok/s  rssΔ(MB)  state(MB)
cosmic     0.1185  2.99    589.24    514.4     756.4       0.22
adam      0.1258  2.33    687.83    440.7    1378.8     624.94
```

Notes:
- `state(MB)` reports tensor state in `optimizer.state`.
- `rssΔ(MB)` is the process RSS delta during the run (includes model + grads + caches).
- The benchmark script's `rss_peak(MB)` column reports the peak RSS observed.

## CPU-only guardrails
CUDA tensors are not supported. Passing CUDA tensors to `Cosmic` or the extension will raise a clear runtime error.

## License
MIT
