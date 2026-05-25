"""Stage: Task Vectors — unlearning via weight arithmetic.

``theta_unlearn = theta_base + (1 - alpha) * (theta_organism - theta_base)``

This is equivalent to:
``theta_unlearn = theta_organism - alpha * (theta_organism - theta_base)``

For each alpha, the interpolated model is saved locally, then uploaded back to
Tinker for evaluation via the dose-response pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, save_info

logger = logging.getLogger(__name__)

TINKER_CLI = shutil.which("tinker") or os.path.join(
    os.path.dirname(sys.executable), "tinker",
)

MERGE_SCRIPT = os.path.join(
    cfg.PROJECT_DIR, "tinker-cookbook",
    "tinker_cookbook", "scripts", "merge_tinker_adapter_to_hf_model.py",
)


def _download_tinker_checkpoint(tinker_path: str, output_dir: str) -> str:
    """Download a Tinker checkpoint to *output_dir*. Returns the local directory."""
    logger.info("  Downloading %s -> %s", tinker_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        [TINKER_CLI, "checkpoint", "download", tinker_path, "--output", output_dir, "--force"],
        check=True,
    )
    adapter_name = tinker_path.replace("tinker://", "").replace("/", "_")
    adapter_dir = os.path.join(output_dir, adapter_name)
    if os.path.isdir(adapter_dir):
        return adapter_dir
    for entry in os.scandir(output_dir):
        if entry.is_dir():
            return entry.path
    return output_dir


def _merge_adapter_to_hf(adapter_dir: str, base_model: str, output_dir: str) -> str:
    """Merge a Tinker LoRA adapter into the HF base model weights."""
    logger.info("  Merging adapter %s + %s -> %s", adapter_dir, base_model, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        [
            sys.executable, MERGE_SCRIPT,
            "--hf-model", base_model,
            "--tinker-adapter-path", adapter_dir,
            "--output-path", output_dir,
        ],
        check=True,
    )
    return output_dir


def _load_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    """Load a HF-format model's state dict from safetensors (or .bin fallback)."""
    safetensors = sorted(
        f for f in os.listdir(model_dir)
        if f.endswith(".safetensors")
    )
    if safetensors:
        state_dict: dict[str, torch.Tensor] = {}
        for fname in safetensors:
            state_dict.update(load_file(os.path.join(model_dir, fname)))
        return state_dict
    bin_files = sorted(
        f for f in os.listdir(model_dir)
        if f.endswith(".bin")
    )
    if bin_files:
        state_dict = {}
        for fname in bin_files:
            state_dict.update(torch.load(os.path.join(model_dir, fname), map_location="cpu"))
        return state_dict
    raise FileNotFoundError(f"No .safetensors or .bin files in {model_dir}")


def _save_state_dict(state_dict: dict[str, torch.Tensor], output_dir: str) -> None:
    """Save a state dict as safetensors, splitting into 5GB shards."""
    max_shard_bytes = 5 * 1024 * 1024 * 1024
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0
    index_map: dict[str, str] = {}

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        tensor_bytes = tensor.nelement() * tensor.element_size()
        if current_size + tensor_bytes > max_shard_bytes and current_shard:
            fname = f"model-{shard_idx:05d}-of-99999.safetensors"
            save_file(current_shard, os.path.join(output_dir, fname))
            for k in current_shard:
                index_map[k] = fname
            shard_idx += 1
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_bytes

    if current_shard:
        if shard_idx == 0:
            fname = "model.safetensors"
        else:
            fname = f"model-{shard_idx:05d}-of-99999.safetensors"
        save_file(current_shard, os.path.join(output_dir, fname))
        for k in current_shard:
            index_map[k] = fname

    total_shards = shard_idx + (1 if current_shard else 0)
    if total_shards > 1:
        for k, v in index_map.items():
            index_map[k] = v.replace("99999", f"{total_shards:05d}")
        for old_name in set(index_map.values()):
            new_name = old_name.replace("99999", f"{total_shards:05d}")
            old_path = os.path.join(output_dir, old_name.replace(f"{total_shards:05d}", "99999"))
            new_path = os.path.join(output_dir, new_name)
            if old_path != new_path and os.path.exists(old_path):
                os.rename(old_path, new_path)

        index = {
            "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in state_dict.values())},
            "weight_map": index_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)


def compute_task_vector(
    organism_dir: str,
    base_dir: str,
    alphas: list[float],
    output_root: str,
) -> list[dict]:
    """Compute task-vector interpolated models for each alpha.

    Returns list of ``{alpha, output_dir}`` dicts.
    """
    logger.info("  Loading organism weights from %s", organism_dir)
    org_sd = _load_state_dict(organism_dir)
    logger.info("  Loading base weights from %s", base_dir)
    base_sd = _load_state_dict(base_dir)

    shared_keys = sorted(set(org_sd.keys()) & set(base_sd.keys()))
    logger.info("  Shared parameter tensors: %d", len(shared_keys))

    delta: dict[str, torch.Tensor] = {}
    for k in shared_keys:
        delta[k] = org_sd[k].float() - base_sd[k].float()

    results: list[dict] = []
    for alpha in alphas:
        out_dir = os.path.join(output_root, f"tv_alpha_{alpha:.2f}")
        os.makedirs(out_dir, exist_ok=True)

        logger.info("  alpha=%.2f: computing theta_base + (1-alpha)*delta", alpha)
        interpolated: dict[str, torch.Tensor] = {}
        for k in shared_keys:
            interpolated[k] = (base_sd[k].float() + (1.0 - alpha) * delta[k]).to(base_sd[k].dtype)
        for k in set(base_sd.keys()) - set(shared_keys):
            interpolated[k] = base_sd[k]

        _save_state_dict(interpolated, out_dir)

        for cfg_file in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "generation_config.json"):
            src = os.path.join(base_dir, cfg_file)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, cfg_file))

        results.append(dict(alpha=alpha, output_dir=out_dir))
        logger.info("  alpha=%.2f: saved to %s", alpha, out_dir)

    return results


# ── Stage orchestrator ───────────────────────────────────────────────────


async def stage_task_vectors(
    seed: int,
    alphas: list[float] | None = None,
    organism_local_path: str | None = None,
    base_local_path: str | None = None,
) -> dict:
    """Produce task-vector interpolated models for each alpha.

    If *organism_local_path* is not given, downloads the organism from Tinker.
    *base_local_path* defaults to the HuggingFace cache for ``cfg.MODEL_NAME``.
    """
    if alphas is None:
        alphas = cfg.TASK_VECTOR_ALPHAS

    tag = f"{cfg.MODEL_SHORT}_tv_s{seed}"
    org_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"

    if info_exists(tag):
        logger.info("[%s] Already exists, skipping", tag)
        return load_info(tag)

    org = load_info(org_tag)
    logger.info(
        "\n%s\n  Stage: Task Vectors (%s, alphas=%s)\n%s",
        "=" * 60, tag, alphas, "=" * 60,
    )

    work_dir = os.path.join(cfg.TINKER_LOG_DIR, tag)
    os.makedirs(work_dir, exist_ok=True)

    if organism_local_path and os.path.isdir(organism_local_path):
        org_merged_dir = organism_local_path
    else:
        sampler_path = org.get("sampler_path", "")
        if not sampler_path:
            raise RuntimeError(f"No sampler_path in organism info for {org_tag}")
        adapter_dir = _download_tinker_checkpoint(sampler_path, os.path.join(work_dir, "org_adapter"))
        org_merged_dir = _merge_adapter_to_hf(
            adapter_dir, cfg.MODEL_NAME, os.path.join(work_dir, "org_merged"),
        )

    if base_local_path and os.path.isdir(base_local_path):
        base_dir = base_local_path
    else:
        from transformers import AutoModelForCausalLM
        logger.info("  Downloading base model %s via HuggingFace...", cfg.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_NAME, trust_remote_code=True, torch_dtype="auto",
        )
        base_dir = os.path.join(work_dir, "base_model")
        os.makedirs(base_dir, exist_ok=True)
        model.save_pretrained(base_dir, safe_serialization=True)
        del model

    tv_results = compute_task_vector(
        org_merged_dir, base_dir, alphas, os.path.join(work_dir, "interpolated"),
    )

    info = dict(
        stage="unlearn", method="tv", tag=tag, model=cfg.MODEL_NAME, seed=seed,
        organism_tag=org_tag, alphas=alphas,
        models=[dict(alpha=r["alpha"], local_path=r["output_dir"]) for r in tv_results],
    )
    save_info(tag, info)
    logger.info("  Task Vectors complete: %d models produced", len(tv_results))
    return info
