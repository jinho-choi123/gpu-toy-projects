"""Profile every Qwen3.8-27B GDN core during a real checkpoint prefill."""

from __future__ import annotations

import argparse
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

DTYPE = torch.bfloat16
ATTENTION_IMPLEMENTATION = "sdpa"
SUPPORTED_SEQUENCE_LENGTHS = (16_384, 32_768, 65_536)
NUM_GDN_LAYERS = 48
LOGITS_TO_KEEP = 1

EAGER_WARMUP = 1
PROFILER_WARMUP = 1

FALLBACK_THRESHOLD = -10.0
MEMORY_HISTORY_MAX_ENTRIES = 100_000
BYTES_PER_MIB = 1024**2

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROFILES_DIR = PROJECT_DIR / "profiles"
ARTICLE_PATH = PROJECT_DIR / "data" / "war_and_peace.txt"

ARTICLE_PLACEHOLDER = "<QWEN38_PROFILE_ARTICLE>"
ARTICLE_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***"
ARTICLE_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***"


class ProfileBackend(StrEnum):
    """GDN backend selected by a Qwen profiling entrypoint."""

    FLASH_QLA = "flash_qla"
    FLA_TRITON = "fla_triton"


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Runtime-configurable dimensions and execution controls."""

    seq_len: int


def _parse_args(argv: Sequence[str] | None) -> ProfileConfig:
    """Parse and return the profiling configuration."""

    def _positive_int(raw: str) -> int:
        value = int(raw)
        if value <= 0:
            raise argparse.ArgumentTypeError("value must be positive")
        return value

    parser = argparse.ArgumentParser(
        description="Profile all Qwen3.8-27B GDN cores during one checkpoint prefill."
    )
    _ = parser.add_argument(
        "--seq-len",
        type=_positive_int,
        choices=SUPPORTED_SEQUENCE_LENGTHS,
        default=16_384,
        help="Exact total chat-template token count.",
    )
    args = parser.parse_args(argv)
    return ProfileConfig(
        seq_len=args.seq_len,
    )


def _to_mib(num_bytes: int) -> float:
    """Convert bytes to MiB."""
    return num_bytes / BYTES_PER_MIB


def _read_article_body() -> str:
    """Read the novel body while excluding the Gutenberg header and license."""
    if not ARTICLE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing profiling text: {ARTICLE_PATH}. "
            "Download Project Gutenberg ebook 2600 before running."
        )

    raw_text = ARTICLE_PATH.read_text(encoding="utf-8")

    start = raw_text.find(ARTICLE_START_MARKER)
    end = raw_text.find(ARTICLE_END_MARKER)

    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not find the expected Project Gutenberg start/end markers.")

    article_body = raw_text[start + len(ARTICLE_START_MARKER) : end].strip()
    if not article_body:
        raise RuntimeError("The extracted War and Peace body is empty.")

    return article_body


def _build_chat_input_ids(tokenizer: Any, config: ProfileConfig) -> torch.Tensor:
    """Build an exact-length valid Qwen chat from a fixed article prefix."""
    seq_len = config.seq_len

    messages = [
        {
            "role": "user",
            "content": (
                "Read the following excerpt from War and Peace. "
                "Do not summarize it yet.\n\n"
                f"{ARTICLE_PLACEHOLDER}"
            ),
        }
    ]

    rendered_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )

    if not isinstance(rendered_chat, str):
        raise RuntimeError("Expected apply_chat_template() to return text.")

    if rendered_chat.count(ARTICLE_PLACEHOLDER) != 1:
        raise RuntimeError("The article placeholder must occur exactly once in the rendered chat.")

    prefix_text, suffix_text = rendered_chat.split(
        ARTICLE_PLACEHOLDER,
        maxsplit=1,
    )

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    article_ids = tokenizer.encode(
        _read_article_body(),
        add_special_tokens=False,
    )

    article_token_count = seq_len - len(prefix_ids) - len(suffix_ids)

    if article_token_count <= 0:
        raise RuntimeError(f"Chat framing exceeds the requested sequence length: {seq_len}.")

    if len(article_ids) < article_token_count:
        raise RuntimeError(
            "War and Peace does not contain enough tokens for "
            f"T={seq_len}: need {article_token_count}, have {len(article_ids)}."
        )

    token_ids = [
        *prefix_ids,
        *article_ids[:article_token_count],
        *suffix_ids,
    ]

    if len(token_ids) != seq_len:
        raise RuntimeError(f"Constructed {len(token_ids)} tokens; expected exactly {seq_len}.")

    logger.info(
        " ".join(
            [
                f"chat_total_tokens={len(token_ids)}",
                f"chat_prefix_tokens={len(prefix_ids)}",
                f"chat_article_tokens={article_token_count}",
                f"chat_suffix_tokens={len(suffix_ids)}",
                "enable_thinking=true",
                "reasoning_effort=xhigh",
            ]
        )
    )

    return torch.tensor([token_ids], dtype=torch.long)


def _run_prefill(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Any:
    """Run one complete checkpoint prefill without generation."""
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=LOGITS_TO_KEEP,
        return_dict=True,
    )


def _memory_snapshot_path(
    backend: ProfileBackend,
    config: ProfileConfig,
) -> Path:
    """Return the full-prefill memory snapshot path."""
    return PROFILES_DIR / (
        f"qwen38_27b_{backend.value}_b1_t{config.seq_len}_memory_snapshot.pickle"
    )


def _profile_eager_forward_memory(
    backend: ProfileBackend,
    config: ProfileConfig,
    device: torch.device,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Record one warmed-up full-prefill CUDA memory snapshot."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = _memory_snapshot_path(backend, config)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)

    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
        max_entries=MEMORY_HISTORY_MAX_ENTRIES,
        device=device,
        clear_history=True,
    )

    try:
        memory_outputs = _run_prefill(
            model,
            input_ids,
            attention_mask,
        )
        torch.cuda.synchronize(device)

        allocated_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        # Record the cache/logits deallocation events in the snapshot.
        del memory_outputs
        torch.cuda.synchronize(device)

        torch.cuda.memory._dump_snapshot(str(snapshot_path))
    finally:
        # Memory recording must end before torch.cuda.profiler.start().
        torch.cuda.memory._record_memory_history(
            enabled=None,
            device=device,
        )

    logger.info(
        " ".join(
            [
                f"memory_backend={backend.value}",
                "memory_batch_size=1",
                f"memory_seq_len={config.seq_len}",
                f"memory_baseline_allocated_mib={_to_mib(allocated_before):.2f}",
                f"memory_after_prefill_allocated_mib={_to_mib(allocated_after):.2f}",
                f"memory_peak_allocated_mib={_to_mib(peak_allocated):.2f}",
                (f"memory_prefill_peak_delta_mib={_to_mib(peak_allocated - allocated_before):.2f}"),
                (
                    "memory_prefill_retained_delta_mib="
                    f"{_to_mib(allocated_after - allocated_before):.2f}"
                ),
                (f"memory_temporary_over_end_mib={_to_mib(peak_allocated - allocated_after):.2f}"),
                f"memory_baseline_reserved_mib={_to_mib(reserved_before):.2f}",
                f"memory_after_prefill_reserved_mib={_to_mib(reserved_after):.2f}",
                f"memory_peak_reserved_mib={_to_mib(peak_reserved):.2f}",
                (f"memory_peak_reserved_delta_mib={_to_mib(peak_reserved - reserved_before):.2f}"),
                f"memory_snapshot={snapshot_path}",
            ]
        )
    )

    # Profiler warmup will repopulate any allocator cache released here.
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


@contextmanager
def _instrument_gdn_calls(
    backend: ProfileBackend,
    gdn_layer_indices: list[int],
) -> Generator[tuple[list[int], dict[int, torch.Tensor]]]:
    """Add per-layer NVTX ranges and capture FlashQLA fallback tensors."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    original_core = modeling_qwen3_5.torch_chunk_gated_delta_rule

    profiled_layers: list[int] = []
    fallback_by_layer: dict[int, torch.Tensor] = {}
    current_decoder_layer: int | None = None

    @wraps(original_core)
    def wrapped_core(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: Any,
    ) -> Any:
        nonlocal current_decoder_layer

        gdn_ordinal = len(profiled_layers)
        if gdn_ordinal >= len(gdn_layer_indices):
            raise RuntimeError("Observed more GDN calls than the model configuration contains.")

        decoder_layer = gdn_layer_indices[gdn_ordinal]
        nvtx_label = f"qwen38_gdn_decoder_layer_{decoder_layer:02d}_gdn_ordinal_{gdn_ordinal:02d}"

        current_decoder_layer = decoder_layer
        try:
            with torch.cuda.nvtx.range(nvtx_label):
                result = original_core(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    chunk_size=chunk_size,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kwargs,
                )
        finally:
            current_decoder_layer = None

        profiled_layers.append(decoder_layer)
        return result

    fallback_module: Any | None = None
    original_get_warmup_chunks_bidi: Any | None = None

    if backend is ProfileBackend.FLASH_QLA:
        from flash_qla.ops.gated_delta_rule.chunk import (
            cp_context as flash_qla_cp_context,
        )

        fallback_module = flash_qla_cp_context
        original_get_warmup_chunks_bidi = flash_qla_cp_context.get_warmup_chunks_bidi

        @wraps(original_get_warmup_chunks_bidi)
        def wrapped_get_warmup_chunks_bidi(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            result = original_get_warmup_chunks_bidi(*args, **kwargs)

            if current_decoder_layer is None:
                raise RuntimeError("FlashQLA fallback_fwd was produced outside a GDN core call.")

            if not isinstance(result, tuple) or len(result) != 4:
                raise RuntimeError("Unexpected get_warmup_chunks_bidi return value.")

            fallback_fwd = result[2]
            if not isinstance(fallback_fwd, torch.Tensor):
                raise RuntimeError("fallback_fwd is not a torch.Tensor.")

            if current_decoder_layer in fallback_by_layer:
                raise RuntimeError(
                    "fallback_fwd was produced more than once for "
                    f"decoder layer {current_decoder_layer}."
                )

            # Keep only the GPU tensor reference here. Copying to CPU would
            # synchronize and contaminate the measured NVTX range.
            fallback_by_layer[current_decoder_layer] = fallback_fwd
            return result

        cast(Any, fallback_module).get_warmup_chunks_bidi = wrapped_get_warmup_chunks_bidi

    modeling_qwen3_5.torch_chunk_gated_delta_rule = wrapped_core

    try:
        yield profiled_layers, fallback_by_layer
    finally:
        modeling_qwen3_5.torch_chunk_gated_delta_rule = original_core

        if fallback_module is not None:
            cast(Any, fallback_module).get_warmup_chunks_bidi = original_get_warmup_chunks_bidi


def _log_fallback_fwd(
    backend: ProfileBackend,
    gdn_layer_indices: list[int],
    fallback_by_layer: dict[int, torch.Tensor],
) -> None:
    """Log FlashQLA fallback tensors and strong/weak head counts."""
    if backend is ProfileBackend.FLA_TRITON:
        logger.info(
            "backend=fla_triton fallback_fwd=not_applicable "
            "strong_decay_head_count=not_applicable "
            "weak_decay_head_count=not_applicable"
        )
        return

    missing_layers = [
        decoder_layer
        for decoder_layer in gdn_layer_indices
        if decoder_layer not in fallback_by_layer
    ]
    if missing_layers:
        raise RuntimeError(
            f"FlashQLA fallback_fwd was not captured for decoder layers {missing_layers}."
        )

    for gdn_ordinal, decoder_layer in enumerate(gdn_layer_indices):
        fallback_cpu = fallback_by_layer[decoder_layer].detach().to(device="cpu")

        if fallback_cpu.ndim != 2:
            raise RuntimeError(
                "Expected fallback_fwd to have shape "
                f"[cp_segments, heads], got {tuple(fallback_cpu.shape)}."
            )

        weak_decay_head_mask = fallback_cpu.any(dim=0)
        weak_decay_head_count = int(weak_decay_head_mask.sum().item())
        strong_decay_head_count = int(fallback_cpu.shape[1]) - weak_decay_head_count

        logger.info(
            " ".join(
                [
                    f"backend={backend.value}",
                    f"decoder_layer={decoder_layer}",
                    f"gdn_ordinal={gdn_ordinal}",
                    f"fallback_threshold={FALLBACK_THRESHOLD}",
                    f"fallback_fwd_shape={tuple(fallback_cpu.shape)}",
                    f"cp_segment_count={fallback_cpu.shape[0]}",
                    (f"fallback_segment_head_count={int(fallback_cpu.sum().item())}"),
                    f"strong_decay_head_count={strong_decay_head_count}",
                    f"weak_decay_head_count={weak_decay_head_count}",
                    (f"weak_decay_head_mask={weak_decay_head_mask.tolist()}"),
                    f"fallback_fwd={fallback_cpu.tolist()}",
                ]
            )
        )


def main(
    backend: ProfileBackend,
    argv: Sequence[str] | None = None,
) -> int:
    """Run the selected Qwen3.8-27B checkpoint prefill profile."""
    config = _parse_args(argv)

    # Keep this assignment even though each entrypoint also sets it.
    # The entrypoint assignment ensures this value is set before FLA import.
    os.environ["FLA_FLASH_QLA"] = "1" if backend is ProfileBackend.FLASH_QLA else "0"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3.8-27B profiling.")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(
            f"Qwen3.8-27B profiling supports SM90 only; found SM{capability[0]}{capability[1]}."
        )

    torch.cuda.set_device(device=device)

    logger.info(
        " ".join(
            [
                f"model_id={MODEL_ID}",
                f"model_revision={MODEL_REVISION}",
                f"backend={backend.value}",
                "batch_size=1",
                f"seq_len={config.seq_len}",
                f"dtype={str(DTYPE).removeprefix('torch.')}",
                f"attention_implementation={ATTENTION_IMPLEMENTATION}",
                "use_cache=true",
                f"logits_to_keep={LOGITS_TO_KEEP}",
                "cuda_graph=false",
                f"eager_warmup={EAGER_WARMUP}",
                "memory_iterations=1",
                f"profiler_warmup={PROFILER_WARMUP}",
                "measured_iterations=1",
                f"device={device}",
                f"device_name={torch.cuda.get_device_name(device)}",
            ]
        )
    )

    # Build the exact chat prompt on CPU before loading the checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    input_ids = _build_chat_input_ids(
        tokenizer,
        config,
    )

    # from_pretrained() downloads missing shards and places all parameters
    # directly on cuda:0 through Accelerate's device-map loading path.
    logger.info(
        "Loading model_id={} revision={} onto {}",
        MODEL_ID,
        MODEL_REVISION,
        device,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=DTYPE,
        device_map=0,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        use_kernels=False,
    )
    model.eval()

    text_config = model.config.get_text_config()
    gdn_layer_indices = [
        layer_index
        for layer_index, layer_type in enumerate(text_config.layer_types)
        if layer_type == "linear_attention"
    ]

    if len(gdn_layer_indices) != NUM_GDN_LAYERS:
        raise RuntimeError(f"Expected {NUM_GDN_LAYERS} GDN layers, found {len(gdn_layer_indices)}.")

    logger.info(
        " ".join(
            [
                f"decoder_layers={text_config.num_hidden_layers}",
                f"gdn_layers={len(gdn_layer_indices)}",
                f"gdn_decoder_layer_indices={gdn_layer_indices}",
                f"logical_qk_heads={text_config.linear_num_key_heads}",
                f"value_heads={text_config.linear_num_value_heads}",
                f"key_head_dim={text_config.linear_key_head_dim}",
                f"value_head_dim={text_config.linear_value_head_dim}",
                f"vocab_size={text_config.vocab_size}",
            ]
        )
    )

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        # Pass 1: model, FLA, Triton, TileLang, and PyTorch SDPA warmup.
        logger.info("phase=eager_warmup iteration=1/1")
        for _ in range(EAGER_WARMUP):
            warmup_outputs = _run_prefill(
                model,
                input_ids,
                attention_mask,
            )
            del warmup_outputs

        torch.cuda.synchronize(device)

        # Pass 2: mandatory full-model eager memory snapshot.
        logger.info("phase=memory iteration=1/1")
        _profile_eager_forward_memory(
            backend,
            config,
            device,
            model,
            input_ids,
            attention_mask,
        )

        # Pass 3: restore the warmed allocator/runtime state after empty_cache().
        logger.info("phase=profiler_warmup iteration=1/1")
        for _ in range(PROFILER_WARMUP):
            profiler_warmup_outputs = _run_prefill(
                model,
                input_ids,
                attention_mask,
            )
            del profiler_warmup_outputs

        torch.cuda.synchronize(device)

        # Pass 4: one measured full prefill. The monkeypatch exists only here.
        logger.info("phase=measured iteration=1/1")
        with _instrument_gdn_calls(
            backend,
            gdn_layer_indices,
        ) as (profiled_layers, fallback_by_layer):
            torch.cuda.profiler.start()
            try:
                measured_outputs = _run_prefill(
                    model,
                    input_ids,
                    attention_mask,
                )
                torch.cuda.synchronize(device)
            finally:
                torch.cuda.profiler.stop()

        if profiled_layers != gdn_layer_indices:
            raise RuntimeError(
                "Measured GDN call order did not match the model config: "
                f"expected {gdn_layer_indices}, got {profiled_layers}."
            )

        _log_fallback_fwd(
            backend,
            gdn_layer_indices,
            fallback_by_layer,
        )

        del measured_outputs
        torch.cuda.synchronize(device)

    return 0
