from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from peft import PeftModel
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from .constants import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_SKILL_ROOT,
)
from .dataset import build_window_dataset, load_skill_spec, load_stats, prepare_skill_directory
from .io_utils import (
    ensure_dir,
    json_ready,
    load_json,
    refresh_symlink,
    remove_if_exists,
    timestamp_run_name,
    utc_now_iso,
    write_json,
    write_yaml,
)
from .lora import build_lora_config, validate_lora_group_targets
from .pi05 import (
    build_processors,
    load_base_policy,
    load_skill_base_policy,
    load_skill_peft_policy,
    make_skill_policy_config,
)
from .schema import SkillSpec

OPTIMIZER_STATE_FILENAME = "optimizer.pt"
SCHEDULER_STATE_FILENAME = "scheduler.pt"
TRAINER_STATE_FILENAME = "trainer_state.json"
RNG_STATE_FILENAME = "rng_state.pt"


@dataclass
class TrainRunConfig:
    skill_id: str
    group: str
    skill_root: Path = DEFAULT_SKILL_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH
    run_name: str | None = None
    steps: int = 1000
    batch_size: int = 4
    eval_every: int = 100
    save_every_steps: int = 2000
    eval_subset_windows: int = 1024
    log_every: int = 10
    num_workers: int = 2
    seed: int = DEFAULT_SEED
    learning_rate: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    decay_steps: int = 30000
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    compile_model: bool = False
    overwrite: bool = False
    train_expert_only: bool = True
    save_last_every_eval: bool = True
    full_eval_at_end: bool = True
    full_eval_log_every_batches: int = 500
    resume_from_run_dir: Path | None = None
    tokenizer_name_or_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _resolved_run_name: str | None = field(default=None, init=False, repr=False)

    def resolve_run_name(self) -> str:
        if self._resolved_run_name is None:
            if self.run_name is not None:
                self._resolved_run_name = self.run_name
            elif self.resume_from_run_dir is not None:
                self._resolved_run_name = Path(self.resume_from_run_dir).name
            else:
                self._resolved_run_name = timestamp_run_name(self.group.lower())
        return self._resolved_run_name

    @property
    def run_dir(self) -> Path:
        if self.resume_from_run_dir is not None:
            return Path(self.resume_from_run_dir)
        return self.output_root / self.skill_id / self.group / self.resolve_run_name()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cycle(loader: DataLoader) -> Any:
    while True:
        for batch in loader:
            yield batch


def _make_loader(dataset: Dataset[dict[str, Any]], cfg: TrainRunConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
    )


def build_eval_monitor_dataset(dataset: Dataset[dict[str, Any]], subset_windows: int, seed: int) -> Dataset[dict[str, Any]]:
    if subset_windows <= 0 or subset_windows >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=subset_windows, replace=False)).tolist()
    return Subset(dataset, indices)


def build_dataloaders(skill_spec: SkillSpec, cfg: TrainRunConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = build_window_dataset(skill_spec, split="train")
    val_dataset = build_window_dataset(skill_spec, split="val")
    if len(train_dataset) == 0:
        raise ValueError(f"skill `{skill_spec.skill_id}` has no train windows.")
    if len(val_dataset) == 0:
        raise ValueError(
            f"skill `{skill_spec.skill_id}` has no val windows. Add more episodes or adjust the split before training."
        )

    val_monitor_dataset = build_eval_monitor_dataset(val_dataset, cfg.eval_subset_windows, cfg.seed)
    train_loader = _make_loader(train_dataset, cfg, shuffle=True)
    val_monitor_loader = _make_loader(val_monitor_dataset, cfg, shuffle=False)
    val_full_loader = _make_loader(val_dataset, cfg, shuffle=False)
    return train_loader, val_monitor_loader, val_full_loader


def _make_scheduler(optimizer: AdamW, *, warmup_steps: int, decay_steps: int) -> LambdaLR:
    decay_steps = max(1, decay_steps)
    warmup_steps = max(0, warmup_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / float(max(1, decay_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")


def save_adapter_checkpoint(model: torch.nn.Module, checkpoint_dir: Path, metadata: dict[str, Any]) -> None:
    tmp_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp"
    remove_if_exists(tmp_dir)
    ensure_dir(tmp_dir)
    model.save_pretrained(str(tmp_dir))
    write_json(tmp_dir / "checkpoint_info.json", metadata)
    remove_if_exists(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def _capture_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: dict[str, Any]) -> None:
    if "python" in payload:
        random.setstate(payload["python"])
    if "numpy" in payload:
        np.random.set_state(payload["numpy"])
    if "torch" in payload:
        torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def is_resumable_checkpoint_dir(checkpoint_dir: Path) -> bool:
    required = [
        checkpoint_dir / TRAINER_STATE_FILENAME,
        checkpoint_dir / OPTIMIZER_STATE_FILENAME,
        checkpoint_dir / SCHEDULER_STATE_FILENAME,
        checkpoint_dir / "adapter_config.json",
    ]
    return all(path.exists() for path in required)


def is_resumable_run_dir(run_dir: Path) -> bool:
    return is_resumable_checkpoint_dir(run_dir / "last")


def save_training_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    *,
    optimizer: AdamW,
    scheduler: LambdaLR,
    trainer_state: dict[str, Any],
) -> None:
    tmp_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp"
    remove_if_exists(tmp_dir)
    ensure_dir(tmp_dir)
    model.save_pretrained(str(tmp_dir))
    write_json(tmp_dir / "checkpoint_info.json", metadata)
    write_json(tmp_dir / TRAINER_STATE_FILENAME, trainer_state)
    torch.save(optimizer.state_dict(), tmp_dir / OPTIMIZER_STATE_FILENAME)
    torch.save(scheduler.state_dict(), tmp_dir / SCHEDULER_STATE_FILENAME)
    torch.save(_capture_rng_state(), tmp_dir / RNG_STATE_FILENAME)
    remove_if_exists(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def load_training_checkpoint(
    checkpoint_dir: Path,
    *,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> dict[str, Any]:
    if not is_resumable_checkpoint_dir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory is not resumable: {checkpoint_dir}")
    optimizer.load_state_dict(torch.load(checkpoint_dir / OPTIMIZER_STATE_FILENAME, map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(checkpoint_dir / SCHEDULER_STATE_FILENAME, map_location="cpu", weights_only=False))
    rng_state_path = checkpoint_dir / RNG_STATE_FILENAME
    if rng_state_path.is_file():
        _restore_rng_state(torch.load(rng_state_path, map_location="cpu", weights_only=False))
    return load_json(checkpoint_dir / TRAINER_STATE_FILENAME)


def evaluate_policy(
    model: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    dataloader: DataLoader,
    *,
    action_dim: int,
    stage: str = "eval",
    log_every_batches: int = 0,
) -> dict[str, float]:
    totals = evaluate_policy_totals(
        model,
        preprocessor,
        postprocessor,
        dataloader,
        action_dim=action_dim,
        stage=stage,
        log_every_batches=log_every_batches,
    )
    return finalize_eval_totals(totals)


def evaluate_policy_totals(
    model: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    dataloader: DataLoader,
    *,
    action_dim: int,
    stage: str = "eval",
    log_every_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_action_mse = 0.0
    total_batches = 0

    with torch.inference_mode():
        total_batches_expected = len(dataloader)
        for batch_index, raw_batch in enumerate(dataloader, start=1):
            proc_batch = preprocessor(raw_batch)
            loss, _ = model(proc_batch)
            pred = model.predict_action_chunk(proc_batch)
            pred = pred[:, :, :action_dim]
            pred = postprocessor(pred)
            target = raw_batch["action"].detach().cpu()
            action_mse = torch.mean((pred - target) ** 2).item()
            total_loss += float(loss.item())
            total_action_mse += action_mse
            total_batches += 1
            if log_every_batches > 0 and (
                batch_index == 1 or batch_index % log_every_batches == 0 or batch_index == total_batches_expected
            ):
                print(
                    f"[{stage}] progress batch={batch_index}/{total_batches_expected}",
                    flush=True,
                )

    if total_batches == 0:
        raise ValueError("Evaluation dataloader produced zero batches.")
    return {
        "loss_sum": total_loss,
        "action_mse_sum": total_action_mse,
        "num_val_batches": float(total_batches),
    }


def finalize_eval_totals(totals: dict[str, float]) -> dict[str, float]:
    total_batches = float(totals["num_val_batches"])
    if total_batches <= 0:
        raise ValueError("Cannot finalize evaluation totals with zero batches.")
    return {
        "val_loss": float(totals["loss_sum"]) / total_batches,
        "action_mse": float(totals["action_mse_sum"]) / total_batches,
        "num_val_batches": total_batches,
    }


def merge_eval_totals(partials: list[dict[str, Any]]) -> dict[str, float]:
    if not partials:
        raise ValueError("Cannot merge empty evaluation partials.")
    merged = {
        "loss_sum": sum(float(item["loss_sum"]) for item in partials),
        "action_mse_sum": sum(float(item["action_mse_sum"]) for item in partials),
        "num_val_batches": float(sum(float(item["num_val_batches"]) for item in partials)),
    }
    merged.update(finalize_eval_totals(merged))
    return merged


def build_batch_shard_subset(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shard_id: int = 0,
    num_shards: int = 1,
) -> tuple[Dataset[dict[str, Any]], dict[str, int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}.")

    total_windows = len(dataset)
    if total_windows <= 0:
        raise ValueError("Cannot shard an empty dataset.")

    total_batches = math.ceil(total_windows / batch_size)
    if num_shards > total_batches:
        raise ValueError(
            f"num_shards={num_shards} exceeds total eval batches={total_batches} for batch_size={batch_size}."
        )

    batch_start = (total_batches * shard_id) // num_shards
    batch_end = (total_batches * (shard_id + 1)) // num_shards
    if batch_end <= batch_start:
        raise ValueError(
            f"Shard {shard_id}/{num_shards} would be empty for total_batches={total_batches} and batch_size={batch_size}."
        )

    window_start = batch_start * batch_size
    window_end = min(total_windows, batch_end * batch_size)
    shard_meta = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "batch_size": batch_size,
        "total_windows": total_windows,
        "total_batches": total_batches,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "shard_num_batches": batch_end - batch_start,
        "window_start": window_start,
        "window_end": window_end,
        "shard_num_windows": window_end - window_start,
    }
    if num_shards == 1:
        return dataset, shard_meta

    indices = list(range(window_start, window_end))
    return Subset(dataset, indices), shard_meta


def build_eval_dataloader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )


def _resolved_config_payload(cfg: TrainRunConfig, skill_spec: SkillSpec, train_windows: int, val_windows: int) -> dict[str, Any]:
    payload = {
        "skill": skill_spec.to_dict(),
        "train": {
            "group": cfg.group,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "eval_every": cfg.eval_every,
            "save_every_steps": cfg.save_every_steps,
            "eval_subset_windows": cfg.eval_subset_windows,
            "log_every": cfg.log_every,
            "num_workers": cfg.num_workers,
            "seed": cfg.seed,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_steps": cfg.warmup_steps,
            "decay_steps": cfg.decay_steps,
            "grad_clip_norm": cfg.grad_clip_norm,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "compile_model": cfg.compile_model,
            "train_expert_only": cfg.train_expert_only,
            "tokenizer_name_or_path": cfg.tokenizer_name_or_path,
            "full_eval_at_end": cfg.full_eval_at_end,
            "full_eval_log_every_batches": cfg.full_eval_log_every_batches,
            "resume_from_run_dir": str(cfg.resume_from_run_dir) if cfg.resume_from_run_dir is not None else None,
            "base_model_path": str(cfg.base_model_path),
            "run_name": cfg.resolve_run_name(),
        },
        "dataset": {
            "train_windows": train_windows,
            "val_windows": val_windows,
        },
        "metadata": cfg.metadata,
    }
    return payload


def update_best_pointer(output_root: Path, run_summary: dict[str, Any]) -> None:
    skill_dir = output_root / run_summary["skill_id"]
    best_meta_path = skill_dir / "best_adapter.json"
    current_metric = run_summary["best_val_loss"]
    if best_meta_path.is_file():
        existing = load_json(best_meta_path)
        if float(existing.get("val_loss", float("inf"))) <= float(current_metric):
            return

    payload = {
        "skill_id": run_summary["skill_id"],
        "group": run_summary["group"],
        "run_name": run_summary["run_name"],
        "adapter_dir": run_summary["best_adapter_dir"],
        "val_loss": run_summary["best_val_loss"],
        "action_mse": run_summary["best_action_mse"],
        "updated_at": utc_now_iso(),
    }
    write_json(best_meta_path, payload)
    refresh_symlink(skill_dir / "best", Path(run_summary["best_adapter_dir"]))


def train_skill_lora(cfg: TrainRunConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    skill_spec = load_skill_spec(cfg.skill_root, cfg.skill_id)
    if not skill_spec.splits_path.is_file() or not skill_spec.stats_path.is_file():
        prepare_skill_directory(skill_spec.skill_dir or (cfg.skill_root / cfg.skill_id), seed=cfg.seed)
    stats_payload = load_stats(skill_spec)
    if skill_spec.source.type == "lerobot":
        print(
            f"[train] building raw datasets skill={cfg.skill_id} "
            f"train_split={skill_spec.splits_path} stats={skill_spec.stats_path}",
            flush=True,
        )
    train_loader, val_monitor_loader, val_full_loader = build_dataloaders(skill_spec, cfg)
    if skill_spec.source.type == "lerobot":
        print(
            f"[train] raw datasets ready skill={cfg.skill_id} "
            f"train_windows={len(train_loader.dataset)} val_windows={len(val_full_loader.dataset)} "
            f"monitor_windows={len(val_monitor_loader.dataset)}",
            flush=True,
        )
    train_iter = _cycle(train_loader)

    run_dir = cfg.run_dir
    resume_requested = cfg.resume_from_run_dir is not None
    resume_checkpoint_dir = run_dir / "last"
    if resume_requested:
        if cfg.run_name is not None and cfg.run_name != run_dir.name:
            raise ValueError(
                f"run_name `{cfg.run_name}` does not match resume run directory name `{run_dir.name}`."
            )
        if cfg.overwrite:
            raise ValueError("Cannot combine resume_from_run_dir with overwrite.")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        if not is_resumable_checkpoint_dir(resume_checkpoint_dir):
            raise FileNotFoundError(
                f"Resume run directory is missing a resumable last checkpoint: {resume_checkpoint_dir}"
            )
    elif run_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    ensure_dir(run_dir)
    print(
        f"[train] start skill={cfg.skill_id} group={cfg.group} run={cfg.resolve_run_name()} "
        f"steps={cfg.steps} batch_size={cfg.batch_size} device={cfg.device} resume={resume_requested}",
        flush=True,
    )

    config = make_skill_policy_config(
        skill_spec,
        base_model_path=cfg.base_model_path,
        device=cfg.device,
        dtype=cfg.dtype,
        train_expert_only=cfg.train_expert_only,
        gradient_checkpointing=cfg.gradient_checkpointing,
        compile_model=cfg.compile_model,
    )
    base_policy = load_base_policy(config, base_model_path=cfg.base_model_path, strict=True)
    validate_lora_group_targets(base_policy, cfg.group)
    if resume_requested:
        policy = PeftModel.from_pretrained(base_policy, str(resume_checkpoint_dir), is_trainable=True)
    else:
        policy = base_policy.wrap_with_peft(
            peft_config=build_lora_config(
                cfg.group,
                base_model_name_or_path=str(cfg.base_model_path),
                inference_mode=False,
            )
        )
    preprocessor, postprocessor = build_processors(
        skill_spec,
        stats_payload,
        device=cfg.device,
        tokenizer_name_or_path=cfg.tokenizer_name_or_path,
    )

    params = [param for param in policy.parameters() if param.requires_grad]
    trainable_params = _count_trainable_params(policy)
    optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    scheduler = _make_scheduler(optimizer, warmup_steps=cfg.warmup_steps, decay_steps=max(cfg.steps, cfg.decay_steps))

    write_yaml(
        run_dir / "resolved_config.yaml",
        _resolved_config_payload(cfg, skill_spec, len(train_loader.dataset), len(val_full_loader.dataset)),
    )

    metrics_log_path = run_dir / "metrics.jsonl"
    monitor_best_val_loss = float("inf")
    monitor_best_action_mse = float("inf")
    monitor_best_step = 0
    resumed_from_step = 0
    best_subset_dir = run_dir / "best_subset"
    last_dir = run_dir / "last"
    best_dir = run_dir / "best"

    if resume_requested:
        resume_state = load_training_checkpoint(
            resume_checkpoint_dir,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        resumed_from_step = int(resume_state.get("step", 0))
        monitor_best_step = int(resume_state.get("monitor_best_step", 0))
        monitor_best_val_loss = float(resume_state.get("monitor_best_val_loss", monitor_best_val_loss))
        monitor_best_action_mse = float(resume_state.get("monitor_best_action_mse", monitor_best_action_mse))
        if resumed_from_step >= cfg.steps:
            raise ValueError(
                f"Resume checkpoint step {resumed_from_step} is already >= requested total steps {cfg.steps}."
            )
        print(
            f"[train] resume skill={cfg.skill_id} group={cfg.group} run={cfg.resolve_run_name()} "
            f"from_step={resumed_from_step}",
            flush=True,
        )
        _append_metrics(
            metrics_log_path,
            {
                "kind": "resume",
                "step": resumed_from_step,
                "resumed_at": utc_now_iso(),
            },
        )

    def current_trainer_state(current_step: int) -> dict[str, Any]:
        return {
            "skill_id": skill_spec.skill_id,
            "group": cfg.group,
            "run_name": cfg.resolve_run_name(),
            "step": current_step,
            "monitor_best_step": monitor_best_step,
            "monitor_best_val_loss": monitor_best_val_loss,
            "monitor_best_action_mse": monitor_best_action_mse,
            "save_every_steps": cfg.save_every_steps,
            "eval_every": cfg.eval_every,
            "batch_size": cfg.batch_size,
            "saved_at": utc_now_iso(),
        }

    def checkpoint_metadata(
        current_step: int,
        *,
        checkpoint_role: str,
        checkpoint_reason: str,
        eval_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "skill_id": skill_spec.skill_id,
            "group": cfg.group,
            "run_name": cfg.resolve_run_name(),
            "step": current_step,
            "checkpoint_role": checkpoint_role,
            "checkpoint_reason": checkpoint_reason,
            "monitor_best_step": monitor_best_step,
            "monitor_best_val_loss": monitor_best_val_loss,
            "monitor_best_action_mse": monitor_best_action_mse,
            "base_model_path": str(cfg.base_model_path),
            "stats_path": str(skill_spec.stats_path),
            "skill_dir": str(skill_spec.skill_dir),
        }
        if eval_metrics is not None:
            payload["monitor_scope"] = eval_metrics["eval_scope"]
            payload["monitor_val_loss"] = eval_metrics["val_loss"]
            payload["monitor_action_mse"] = eval_metrics["action_mse"]
        return payload

    for step in range(resumed_from_step + 1, cfg.steps + 1):
        raw_batch = next(train_iter)
        proc_batch = preprocessor(raw_batch)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy(proc_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0 or step == 1 or step == cfg.steps:
            print(
                f"[train] skill={cfg.skill_id} group={cfg.group} step={step}/{cfg.steps} "
                f"loss={float(loss.item()):.6f} lr={float(optimizer.param_groups[0]['lr']):.6e}",
                flush=True,
            )
            _append_metrics(
                metrics_log_path,
                {
                    "kind": "train",
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "timestamp": utc_now_iso(),
                },
            )

        eval_metrics: dict[str, Any] | None = None
        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = evaluate_policy(
                policy,
                preprocessor,
                postprocessor,
                val_monitor_loader,
                action_dim=skill_spec.action_dim,
                stage="eval_subset",
                log_every_batches=0,
            )
            metrics["step"] = step
            metrics["kind"] = "eval_subset"
            metrics["eval_scope"] = "subset" if len(val_monitor_loader.dataset) < len(val_full_loader.dataset) else "full"
            metrics["timestamp"] = utc_now_iso()
            print(
                f"[eval_subset] skill={cfg.skill_id} group={cfg.group} step={step}/{cfg.steps} "
                f"val_loss={metrics['val_loss']:.6f} action_mse={metrics['action_mse']:.6f}",
                flush=True,
            )
            _append_metrics(metrics_log_path, metrics)
            eval_metrics = metrics

            if metrics["val_loss"] < monitor_best_val_loss:
                monitor_best_val_loss = metrics["val_loss"]
                monitor_best_action_mse = metrics["action_mse"]
                monitor_best_step = step
                save_adapter_checkpoint(
                    policy,
                    best_subset_dir,
                    checkpoint_metadata(
                        step,
                        checkpoint_role="best_subset",
                        checkpoint_reason="monitor_improved",
                        eval_metrics=metrics,
                    ),
                )

        should_save_periodic = cfg.save_every_steps > 0 and step % cfg.save_every_steps == 0
        should_save_eval = eval_metrics is not None and cfg.save_last_every_eval
        should_save_final = step == cfg.steps
        if should_save_periodic or should_save_eval or should_save_final:
            save_reasons = []
            if should_save_periodic:
                save_reasons.append("periodic")
            if should_save_eval:
                save_reasons.append("eval")
            if should_save_final:
                save_reasons.append("final")
            save_training_checkpoint(
                policy,
                last_dir,
                checkpoint_metadata(
                    step,
                    checkpoint_role="last",
                    checkpoint_reason="+".join(save_reasons),
                    eval_metrics=eval_metrics,
                ),
                optimizer=optimizer,
                scheduler=scheduler,
                trainer_state=current_trainer_state(step),
            )

    full_eval_candidates: list[tuple[str, Path, int]] = []
    if cfg.full_eval_at_end:
        full_eval_candidates.append(("last", last_dir, cfg.steps))
        if best_subset_dir.is_dir() and monitor_best_step not in {0, cfg.steps}:
            full_eval_candidates.append(("best_subset", best_subset_dir, monitor_best_step))

    del optimizer
    del scheduler
    del params
    del policy
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    full_eval_results: list[dict[str, Any]] = []
    for label, adapter_dir, candidate_step in full_eval_candidates:
        print(
            f"[eval_full] skill={cfg.skill_id} group={cfg.group} checkpoint={label} step={candidate_step} begin",
            flush=True,
        )
        model = load_skill_peft_policy(
            skill_spec,
            adapter_dir,
            base_model_path=cfg.base_model_path,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        metrics = evaluate_policy(
            model,
            preprocessor,
            postprocessor,
            val_full_loader,
            action_dim=skill_spec.action_dim,
            stage=f"eval_full_{label}",
            log_every_batches=cfg.full_eval_log_every_batches,
        )
        metrics["step"] = candidate_step
        metrics["kind"] = "eval_full"
        metrics["checkpoint"] = label
        metrics["eval_scope"] = "full"
        metrics["timestamp"] = utc_now_iso()
        print(
            f"[eval_full] skill={cfg.skill_id} group={cfg.group} checkpoint={label} step={candidate_step} "
            f"val_loss={metrics['val_loss']:.6f} action_mse={metrics['action_mse']:.6f}",
            flush=True,
        )
        _append_metrics(metrics_log_path, metrics)
        full_eval_results.append(
            {
                "checkpoint": label,
                "adapter_dir": str(adapter_dir),
                "step": candidate_step,
                **metrics,
            }
        )
        del model
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if full_eval_results:
        selected = min(full_eval_results, key=lambda item: float(item["val_loss"]))
    else:
        selected_label = "best_subset" if best_subset_dir.is_dir() else "last"
        selected_adapter_dir = best_subset_dir if selected_label == "best_subset" else last_dir
        selected_step = monitor_best_step if selected_label == "best_subset" else cfg.steps
        selected = {
            "checkpoint": selected_label,
            "adapter_dir": str(selected_adapter_dir),
            "step": selected_step,
            "val_loss": float(monitor_best_val_loss),
            "action_mse": float(monitor_best_action_mse),
            "num_val_batches": float(len(val_monitor_loader)),
            "kind": "eval_subset",
            "eval_scope": "subset",
            "timestamp": utc_now_iso(),
        }

    remove_if_exists(best_dir)
    shutil.copytree(Path(selected["adapter_dir"]), best_dir)
    best_checkpoint_meta = {
        "skill_id": skill_spec.skill_id,
        "group": cfg.group,
        "run_name": cfg.resolve_run_name(),
        "step": int(selected["step"]),
        "val_loss": float(selected["val_loss"]),
        "action_mse": float(selected["action_mse"]),
        "selected_from": str(selected["checkpoint"]),
        "base_model_path": str(cfg.base_model_path),
        "stats_path": str(skill_spec.stats_path),
        "skill_dir": str(skill_spec.skill_dir),
    }
    write_json(best_dir / "checkpoint_info.json", best_checkpoint_meta)

    summary = {
        "skill_id": skill_spec.skill_id,
        "group": cfg.group,
        "run_name": cfg.resolve_run_name(),
        "run_dir": str(run_dir),
        "best_step": int(selected["step"]),
        "best_val_loss": float(selected["val_loss"]),
        "best_action_mse": float(selected["action_mse"]),
        "resumed_from_step": resumed_from_step,
        "resume_from_run_dir": str(cfg.resume_from_run_dir) if cfg.resume_from_run_dir is not None else None,
        "best_checkpoint": str(selected["checkpoint"]),
        "best_adapter_dir": str(best_dir),
        "last_adapter_dir": str(last_dir),
        "best_subset_adapter_dir": str(best_subset_dir),
        "monitor_best_step": monitor_best_step,
        "monitor_best_val_loss": monitor_best_val_loss,
        "monitor_best_action_mse": monitor_best_action_mse,
        "save_every_steps": cfg.save_every_steps,
        "monitor_eval_windows": len(val_monitor_loader.dataset),
        "train_windows": len(train_loader.dataset),
        "val_windows": len(val_full_loader.dataset),
        "full_eval_candidates": full_eval_results,
        "trainable_params": trainable_params,
        "completed_at": utc_now_iso(),
    }
    write_json(run_dir / "summary.json", summary)
    update_best_pointer(cfg.output_root, summary)
    print(f"[train] done summary={summary}", flush=True)
    return summary


def evaluate_saved_adapter(
    *,
    skill_id: str,
    skill_root: Path = DEFAULT_SKILL_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    adapter_dir: Path | None = None,
    group: str | None = None,
    run_name: str | None = None,
    batch_size: int = 4,
    num_workers: int = 2,
    device: str = "cuda",
    dtype: str = "bfloat16",
    tokenizer_name_or_path: str | None = None,
    write_result: bool = True,
) -> dict[str, Any]:
    from .registry import SkillAdapterRegistry

    registry = SkillAdapterRegistry(
        skill_root=skill_root,
        output_root=output_root,
        base_model_path=base_model_path,
        device=device,
        dtype=dtype,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    skill_spec = load_skill_spec(skill_root, skill_id)
    adapter_ref = registry.resolve_adapter_reference(
        skill_id,
        group=group,
        run_name=run_name,
        adapter_dir=adapter_dir,
    )

    model = load_skill_peft_policy(
        skill_spec,
        adapter_ref.adapter_dir,
        base_model_path=base_model_path,
        device=device,
        dtype=dtype,
    )
    preprocessor, postprocessor = build_processors(
        skill_spec,
        load_stats(skill_spec),
        device=device,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    val_dataset = build_window_dataset(skill_spec, split="val")
    if len(val_dataset) == 0:
        raise ValueError(f"skill `{skill_id}` has no val windows.")
    val_loader = build_eval_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    metrics = evaluate_policy(
        model,
        preprocessor,
        postprocessor,
        val_loader,
        action_dim=skill_spec.action_dim,
        stage="eval_saved_adapter",
        log_every_batches=500,
    )
    payload = {
        "skill_id": skill_id,
        "model_type": "adapter",
        "group": adapter_ref.group,
        "run_name": adapter_ref.run_name,
        "adapter_dir": str(adapter_ref.adapter_dir),
        "base_model_path": str(base_model_path),
        **metrics,
        "evaluated_at": utc_now_iso(),
    }
    eval_summary_path = adapter_ref.adapter_dir / "eval_summary.json"
    payload["eval_summary_path"] = str(eval_summary_path)
    if write_result:
        write_json(eval_summary_path, payload)
    return payload


def evaluate_base_policy(
    *,
    skill_id: str,
    skill_root: Path = DEFAULT_SKILL_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    batch_size: int = 4,
    num_workers: int = 2,
    device: str = "cuda",
    dtype: str = "bfloat16",
    tokenizer_name_or_path: str | None = None,
    write_result: bool = True,
) -> dict[str, Any]:
    skill_spec = load_skill_spec(skill_root, skill_id)
    model = load_skill_base_policy(
        skill_spec,
        base_model_path=base_model_path,
        device=device,
        dtype=dtype,
    )
    preprocessor, postprocessor = build_processors(
        skill_spec,
        load_stats(skill_spec),
        device=device,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    val_dataset = build_window_dataset(skill_spec, split="val")
    if len(val_dataset) == 0:
        raise ValueError(f"skill `{skill_id}` has no val windows.")
    val_loader = build_eval_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    metrics = evaluate_policy(
        model,
        preprocessor,
        postprocessor,
        val_loader,
        action_dim=skill_spec.action_dim,
        stage="eval_base",
        log_every_batches=500,
    )
    eval_summary_path = output_root / skill_id / "base_eval_summary.json"
    payload = {
        "skill_id": skill_id,
        "model_type": "base",
        "group": None,
        "run_name": None,
        "adapter_dir": None,
        "base_model_path": str(base_model_path),
        **metrics,
        "evaluated_at": utc_now_iso(),
        "eval_summary_path": str(eval_summary_path),
    }
    if write_result:
        write_json(eval_summary_path, payload)
    return payload


def evaluate_base_policy_shard(
    *,
    skill_id: str,
    shard_id: int,
    num_shards: int,
    skill_root: Path = DEFAULT_SKILL_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    batch_size: int = 4,
    num_workers: int = 2,
    device: str = "cuda",
    dtype: str = "bfloat16",
    tokenizer_name_or_path: str | None = None,
    log_every_batches: int = 500,
    shard_output_path: Path | None = None,
    write_result: bool = True,
) -> dict[str, Any]:
    skill_spec = load_skill_spec(skill_root, skill_id)
    model = load_skill_base_policy(
        skill_spec,
        base_model_path=base_model_path,
        device=device,
        dtype=dtype,
    )
    preprocessor, postprocessor = build_processors(
        skill_spec,
        load_stats(skill_spec),
        device=device,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    val_dataset = build_window_dataset(skill_spec, split="val")
    if len(val_dataset) == 0:
        raise ValueError(f"skill `{skill_id}` has no val windows.")
    shard_dataset, shard_meta = build_batch_shard_subset(
        val_dataset,
        batch_size=batch_size,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    val_loader = build_eval_dataloader(
        shard_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    totals = evaluate_policy_totals(
        model,
        preprocessor,
        postprocessor,
        val_loader,
        action_dim=skill_spec.action_dim,
        stage=f"eval_base_shard_{shard_id}",
        log_every_batches=log_every_batches,
    )
    metrics = finalize_eval_totals(totals)
    if shard_output_path is None:
        shard_output_path = output_root / skill_id / "base_eval_shards" / f"shard_{shard_id:02d}_of_{num_shards:02d}.json"
    payload = {
        "skill_id": skill_id,
        "model_type": "base",
        "group": None,
        "run_name": None,
        "adapter_dir": None,
        "base_model_path": str(base_model_path),
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "num_workers": num_workers,
        **shard_meta,
        **totals,
        **metrics,
        "evaluated_at": utc_now_iso(),
        "shard_output_path": str(shard_output_path),
    }
    if write_result:
        write_json(shard_output_path, payload)
    return payload


def merge_base_eval_shards(
    shard_payloads: list[dict[str, Any]],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    write_result: bool = True,
) -> dict[str, Any]:
    if not shard_payloads:
        raise ValueError("Cannot merge empty base evaluation shards.")
    ordered = sorted(shard_payloads, key=lambda item: int(item["shard_id"]))
    first = ordered[0]
    skill_id = str(first["skill_id"])
    num_shards = int(first["num_shards"])
    seen_ids = {int(item["shard_id"]) for item in ordered}
    expected_ids = set(range(num_shards))
    if seen_ids != expected_ids:
        raise ValueError(f"Expected shard ids {sorted(expected_ids)}, got {sorted(seen_ids)}.")
    for item in ordered:
        if str(item["skill_id"]) != skill_id:
            raise ValueError("All shard payloads must belong to the same skill.")
        if int(item["num_shards"]) != num_shards:
            raise ValueError("All shard payloads must agree on num_shards.")
        if str(item["model_type"]) != "base":
            raise ValueError("merge_base_eval_shards only accepts base-model shard payloads.")

    merged = merge_eval_totals(ordered)
    eval_summary_path = output_root / skill_id / "base_eval_summary.json"
    payload = {
        "skill_id": skill_id,
        "model_type": "base",
        "group": None,
        "run_name": None,
        "adapter_dir": None,
        "base_model_path": str(first["base_model_path"]),
        "evaluation_mode": "sharded" if num_shards > 1 else "single",
        "num_shards": num_shards,
        "total_val_windows": int(sum(int(item["shard_num_windows"]) for item in ordered)),
        "loss_sum": float(merged["loss_sum"]),
        "action_mse_sum": float(merged["action_mse_sum"]),
        "num_val_batches": float(merged["num_val_batches"]),
        "val_loss": float(merged["val_loss"]),
        "action_mse": float(merged["action_mse"]),
        "shards": [
            {
                "shard_id": int(item["shard_id"]),
                "device": str(item["device"]),
                "shard_num_windows": int(item["shard_num_windows"]),
                "shard_num_batches": int(item["shard_num_batches"]),
                "loss_sum": float(item["loss_sum"]),
                "action_mse_sum": float(item["action_mse_sum"]),
                "val_loss": float(item["val_loss"]),
                "action_mse": float(item["action_mse"]),
                "shard_output_path": str(item["shard_output_path"]),
            }
            for item in ordered
        ],
        "evaluated_at": utc_now_iso(),
        "eval_summary_path": str(eval_summary_path),
    }
    if write_result:
        write_json(eval_summary_path, payload)
    return payload


def compare_eval_results(adapter_summary: dict[str, Any], base_summary: dict[str, Any]) -> dict[str, Any]:
    if adapter_summary["skill_id"] != base_summary["skill_id"]:
        raise ValueError(
            f"Cannot compare different skills: `{adapter_summary['skill_id']}` vs `{base_summary['skill_id']}`."
        )
    adapter_val_loss = float(adapter_summary["val_loss"])
    adapter_action_mse = float(adapter_summary["action_mse"])
    base_val_loss = float(base_summary["val_loss"])
    base_action_mse = float(base_summary["action_mse"])
    return {
        "skill_id": adapter_summary["skill_id"],
        "adapter": adapter_summary,
        "base": base_summary,
        "improvement_over_base": {
            "val_loss": base_val_loss - adapter_val_loss,
            "action_mse": base_action_mse - adapter_action_mse,
        },
        "better_than_base": {
            "val_loss": adapter_val_loss < base_val_loss,
            "action_mse": adapter_action_mse < base_action_mse,
        },
        "compared_at": utc_now_iso(),
    }
