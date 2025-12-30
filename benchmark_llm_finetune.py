from __future__ import annotations

import copy
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from cosmic import Cosmic


def _disable_torchvision() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    try:
        from transformers.utils import import_utils
    except Exception:
        return
    import_utils._torchvision_available = False
    import_utils._torchvision_version = "N/A"


def _read_rss_kb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def _read_cpu_times() -> tuple[int, int]:
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("cpu "):
                    parts = line.split()
                    values = [int(value) for value in parts[1:]]
                    total = sum(values)
                    idle = values[3] if len(values) > 3 else 0
                    if len(values) > 4:
                        idle += values[4]
                    return total, idle
    except FileNotFoundError:
        pass
    return 0, 0


def _build_dataset(model_name: str, block_size: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer, default_data_collator

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {}
        for key, tokens in concatenated.items():
            tokens = tokens[:total_len]
            result[key] = [tokens[i : i + block_size] for i in range(0, total_len, block_size)]
        result["labels"] = list(result["input_ids"])
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)
    lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return lm_dataset, default_data_collator


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    state_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                state_bytes += value.numel() * value.element_size()
    return state_bytes


def _run_optimizer(
    name: str,
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    dataset,
    collate_fn,
    batch_size: int,
    num_steps: int,
    lr: float,
    weight_decay: float,
    log_stream,
) -> dict[str, float]:
    torch.manual_seed(0)
    model.load_state_dict(base_state)
    model.train()

    generator = torch.Generator().manual_seed(0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )

    if name == "cosmic":
        optimizer = Cosmic(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    total_tokens = 0
    loss = torch.tensor(0.0)
    loss_sum = 0.0
    loss_count = 0
    data_iter = iter(dataloader)
    start = time.perf_counter()
    cpu_start = time.process_time()
    sys_cpu_start, sys_cpu_idle_start = _read_cpu_times()
    start_rss = _read_rss_kb()
    peak_rss = start_rss

    for step in range(1, num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        outputs = model(**batch)
        loss = outputs.loss
        loss_val = float(loss.item())
        loss_sum += loss_val
        loss_count += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += int(batch["input_ids"].numel())
        if step % 50 == 0:
            if log_stream is None:
                print(f"{name} step={step} loss={loss_val:.4f}")
            else:
                print(f"{name} step={step} loss={loss_val:.4f}", file=log_stream)

        rss_kb = _read_rss_kb()
        if rss_kb:
            peak_rss = max(peak_rss, rss_kb)

    elapsed = time.perf_counter() - start
    cpu_time = time.process_time() - cpu_start
    sys_cpu_end, sys_cpu_idle_end = _read_cpu_times()
    sys_cpu_util = 0.0
    if sys_cpu_end > sys_cpu_start:
        total_delta = sys_cpu_end - sys_cpu_start
        idle_delta = sys_cpu_idle_end - sys_cpu_idle_start
        if total_delta > 0:
            sys_cpu_util = (total_delta - idle_delta) / total_delta * 100.0
    tokens_per_s = total_tokens / elapsed if elapsed > 0 else 0.0
    steps_per_s = num_steps / elapsed if elapsed > 0 else 0.0
    state_bytes = _optimizer_state_bytes(optimizer)
    avg_loss = loss_sum / loss_count if loss_count else float(loss.item())
    perplexity = math.exp(min(avg_loss, 50.0))

    return {
        "final_loss": float(loss.item()),
        "avg_loss": float(avg_loss),
        "perplexity": float(perplexity),
        "elapsed_s": elapsed,
        "tokens_per_s": tokens_per_s,
        "steps_per_s": steps_per_s,
        "total_tokens": float(total_tokens),
        "cpu_time_s": float(cpu_time),
        "cpu_util_percent": float((cpu_time / elapsed) * 100.0) if elapsed > 0 else 0.0,
        "sys_cpu_util_percent": float(sys_cpu_util),
        "peak_rss_mb": float(peak_rss) / 1024.0,
        "rss_delta_mb": float(max(0, peak_rss - start_rss)) / 1024.0,
        "optimizer_state_mb": float(state_bytes) / (1024.0 * 1024.0),
    }


def main() -> None:
    torch.manual_seed(0)

    model_name = "distilgpt2"
    batch_size = 4
    block_size = 128
    num_steps = 600
    lr = 5e-4
    weight_decay = 0.01

    _disable_torchvision()
    from transformers import AutoModelForCausalLM

    dataset, collate_fn = _build_dataset(model_name, block_size)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    base_state = copy.deepcopy(model.state_dict())
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    optimizers = ["cosmic", "adam"]
    results: dict[str, dict[str, float]] = {}
    for name in optimizers:
        results[name] = _run_optimizer(
            name,
            model,
            base_state,
            dataset,
            collate_fn,
            batch_size,
            num_steps,
            lr,
            weight_decay,
            log_stream=None,
        )

    print("\nSummary")
    print(f"model_params={param_count}")
    header = (
        f"{'optimizer':<10} {'loss':>10} {'ppl':>10} {'time(s)':>10} {'tok/s':>10} "
        f"{'cpu%':>7} {'rss_peak(MB)':>12} {'state(MB)':>11}"
    )
    print(header)
    for name in optimizers:
        metrics = results[name]
        print(
            f"{name:<10}"
            f" {metrics['final_loss']:>10.4f}"
            f" {metrics['perplexity']:>10.2f}"
            f" {metrics['elapsed_s']:>10.2f}"
            f" {metrics['tokens_per_s']:>10.1f}"
            f" {metrics['cpu_util_percent']:>7.1f}"
            f" {metrics['peak_rss_mb']:>12.1f}"
            f" {metrics['optimizer_state_mb']:>11.2f}"
        )
    print("\nDetails")
    for name in optimizers:
        metrics = results[name]
        print(
            f"{name}_steps_per_s={metrics['steps_per_s']:.2f} "
            f"{name}_avg_loss={metrics['avg_loss']:.4f} "
            f"{name}_cpu_util={metrics['cpu_util_percent']:.1f}% "
            f"{name}_sys_cpu_util={metrics['sys_cpu_util_percent']:.1f}% "
            f"{name}_cpu_time_s={metrics['cpu_time_s']:.1f} "
            f"{name}_rss_delta_mb={metrics['rss_delta_mb']:.1f} "
            f"{name}_total_tokens={int(metrics['total_tokens'])}"
        )


if __name__ == "__main__":
    main()
