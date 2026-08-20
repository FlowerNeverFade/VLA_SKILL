#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_skill.constants import DEFAULT_BASE_MODEL_PATH, DEFAULT_OUTPUT_ROOT, DEFAULT_SKILL_ROOT
from vla_skill.dataset import build_single_observation_batch, build_window_dataset, load_obs_json, load_skill_spec, load_stats
from vla_skill.pi05 import build_processors, load_skill_base_policy
from vla_skill.registry import SkillAdapterRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PI05 skill adapter inference on a dataset sample or obs.json.")
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--skill-id", type=str, help="Explicit adapter skill_id override.")
    parser.add_argument("--task", type=str, help="Task text for router selection.")
    parser.add_argument("--group", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--obs-json", type=Path, help="External observation JSON.")
    parser.add_argument("--sample-skill-id", type=str, help="Skill directory to source a dataset sample from.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--sample-index", type=int, help="Window index inside the split dataset.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _load_dataset_sample(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if not args.sample_skill_id or args.sample_index is None:
        raise SystemExit("Dataset mode requires --sample-skill-id and --sample-index.")
    skill_spec = load_skill_spec(args.skill_root, args.sample_skill_id)
    dataset = build_window_dataset(skill_spec, split=args.split)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise SystemExit(f"sample-index {args.sample_index} is out of range for {args.sample_skill_id}/{args.split}.")
    sample = dataset[args.sample_index]
    raw_batch = {key: value for key, value in sample.items() if key != "action"}
    return raw_batch, {
        "source": "dataset",
        "sample_skill_id": args.sample_skill_id,
        "split": args.split,
        "sample_index": args.sample_index,
        "ground_truth_action": sample["action"].tolist(),
        "sample_task": sample["task"],
    }


def _predict_action_payload(
    *,
    model: Any,
    skill_spec: Any,
    preprocessor: Any,
    postprocessor: Any,
    raw_batch: dict[str, Any],
) -> list[list[float]]:
    proc_batch = preprocessor(raw_batch)
    pred = model.predict_action_chunk(proc_batch)
    pred = pred[:, :, : skill_spec.action_dim]
    pred = postprocessor(pred)
    return pred.squeeze(0).tolist()


def _build_base_inference_components(
    *,
    skill_spec: Any,
    base_model_path: Path,
    device: str,
    dtype: str,
    tokenizer_name_or_path: str | None,
) -> tuple[Any, Any, Any]:
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
    return model, preprocessor, postprocessor


def _sample_action_mse(predicted_action: list[list[float]], ground_truth_action: list[list[float]]) -> float:
    pred = np.asarray(predicted_action, dtype=np.float32)
    target = np.asarray(ground_truth_action, dtype=np.float32)
    return float(np.mean((pred - target) ** 2))


def main() -> None:
    args = parse_args()
    if args.base_only and args.compare_base:
        raise SystemExit("--base-only and --compare-base cannot be used together.")
    registry = SkillAdapterRegistry(
        skill_root=args.skill_root,
        output_root=args.output_root,
        base_model_path=args.base_model_path,
        device=args.device,
        dtype=args.dtype,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
    )
    if args.obs_json:
        payload = load_obs_json(args.obs_json)
        explicit_skill_id = args.skill_id or payload.get("skill_id")
        task_text = args.task or payload.get("task")
        if explicit_skill_id is None and not task_text:
            raise SystemExit("obs.json mode requires either --skill-id / skill_id or task text.")
        selected_skill_id = explicit_skill_id or registry.router.resolve(str(task_text)).skill_spec.skill_id
        skill_spec_for_input = load_skill_spec(args.skill_root, selected_skill_id)
        image_paths = payload.get("image_paths") or payload.get("images")
        if not isinstance(image_paths, dict):
            raise SystemExit("obs.json must contain `image_paths` or `images` mapping camera names to image files.")
        raw_batch = build_single_observation_batch(
            skill_spec_for_input,
            task=str(task_text or ""),
            state=payload["state"],
            image_paths=image_paths,
        )
        source_meta = {"source": "obs_json", "selected_skill_id": selected_skill_id}
        adapter_skill_id = selected_skill_id
    else:
        raw_batch, source_meta = _load_dataset_sample(args)
        task_text = args.task or raw_batch.get("task")
        adapter_skill_id = args.skill_id

    if args.base_only:
        if adapter_skill_id is not None:
            skill_spec = load_skill_spec(args.skill_root, adapter_skill_id)
        elif args.obs_json:
            skill_spec = load_skill_spec(args.skill_root, source_meta["selected_skill_id"])
        else:
            skill_spec = load_skill_spec(args.skill_root, args.sample_skill_id)
        model, preprocessor, postprocessor = _build_base_inference_components(
            skill_spec=skill_spec,
            base_model_path=args.base_model_path,
            device=args.device,
            dtype=args.dtype,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
        )
        predicted_action = _predict_action_payload(
            model=model,
            skill_spec=skill_spec,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            raw_batch=raw_batch,
        )
        payload = {
            "skill_id": skill_spec.skill_id,
            "model_type": "base",
            "group": None,
            "run_name": None,
            "adapter_dir": None,
            "task": task_text,
            "predicted_action": predicted_action,
            "source": source_meta,
        }
    else:
        skill_spec, model, preprocessor, postprocessor, adapter_ref = registry.activate(
            skill_id=adapter_skill_id,
            task=task_text,
            group=args.group,
            run_name=args.run_name,
            adapter_dir=args.adapter_dir,
        )
        predicted_action = _predict_action_payload(
            model=model,
            skill_spec=skill_spec,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            raw_batch=raw_batch,
        )
        payload = {
            "skill_id": skill_spec.skill_id,
            "model_type": "adapter",
            "group": adapter_ref.group,
            "run_name": adapter_ref.run_name,
            "adapter_dir": str(adapter_ref.adapter_dir),
            "task": task_text,
            "predicted_action": predicted_action,
            "source": source_meta,
        }

        if args.compare_base:
            base_model, base_preprocessor, base_postprocessor = _build_base_inference_components(
                skill_spec=skill_spec,
                base_model_path=args.base_model_path,
                device=args.device,
                dtype=args.dtype,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
            )
            base_predicted_action = _predict_action_payload(
                model=base_model,
                skill_spec=skill_spec,
                preprocessor=base_preprocessor,
                postprocessor=base_postprocessor,
                raw_batch=raw_batch,
            )
            payload = {
                "skill_id": skill_spec.skill_id,
                "task": task_text,
                "source": source_meta,
                "adapter": payload,
                "base": {
                    "skill_id": skill_spec.skill_id,
                    "model_type": "base",
                    "group": None,
                    "run_name": None,
                    "adapter_dir": None,
                    "task": task_text,
                    "predicted_action": base_predicted_action,
                    "source": source_meta,
                },
            }
            if "ground_truth_action" in source_meta:
                adapter_mse = _sample_action_mse(payload["adapter"]["predicted_action"], source_meta["ground_truth_action"])
                base_mse = _sample_action_mse(base_predicted_action, source_meta["ground_truth_action"])
                payload["sample_action_mse"] = {
                    "adapter": adapter_mse,
                    "base": base_mse,
                    "improvement_over_base": base_mse - adapter_mse,
                }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
