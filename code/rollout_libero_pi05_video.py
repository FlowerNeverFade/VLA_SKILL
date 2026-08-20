#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_libero_compat(*args: Any, **kwargs: Any) -> Any:
    # LIBERO init-state files are local trusted torch pickles. PyTorch 2.6
    # changed the default to weights_only=True, which breaks LIBERO's loader.
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_libero_compat

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from vla_skill.constants import DEFAULT_BASE_MODEL_PATH, DEFAULT_OUTPUT_ROOT, DEFAULT_SKILL_ROOT
from vla_skill.dataset import _image_value_to_tensor, load_skill_spec, load_stats
from vla_skill.pi05 import build_processors, load_skill_base_policy, load_skill_peft_policy


LIBERO_DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
LIBERO_SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class PolicyBundle:
    name: str
    model: Any
    preprocessor: Any
    postprocessor: Any
    device: str


@dataclass
class RolloutResult:
    policy_name: str
    success: bool
    steps: int
    episode_index: int
    seed: int
    frames: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out PI05 base and skill LoRA in LIBERO and save videos.")
    parser.add_argument("--skill-id", default="libero_16_turn_on_the_stove")
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--group", default="A")
    parser.add_argument("--run-name", default="stove16_bg10k_bs8_a")
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=16)
    parser.add_argument("--video-preset", default="slow")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--base-device", default="cuda:0")
    parser.add_argument("--adapter-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT.parent / "rollout_videos")
    parser.add_argument("--save-all-candidates", action="store_true")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Continue through the full requested episode range instead of stopping at the first base-fail/adapter-success case.",
    )
    return parser.parse_args()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def libero_state(raw_obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


def raw_batch_from_obs(raw_obs: dict[str, Any], task: str) -> dict[str, Any]:
    base_img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "task": task,
        "observation.state": torch.from_numpy(libero_state(raw_obs)),
        "observation.images.base_0_rgb": _image_value_to_tensor(base_img),
        "observation.images.left_wrist_0_rgb": _image_value_to_tensor(wrist_img),
        "observation.images.right_wrist_0_rgb": _image_value_to_tensor(wrist_img),
    }


def render_frame(raw_obs: dict[str, Any], *, label: str, success: bool, step: int) -> np.ndarray:
    base_img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]).copy()
    wrist_img = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]).copy()
    h, w = base_img.shape[:2]
    inset_w = max(96, w // 3)
    inset_h = max(96, h // 3)
    margin = max(8, w // 64)
    wrist_small = cv2.resize(wrist_img, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    frame = base_img.copy()
    frame[margin : margin + inset_h, w - inset_w - margin : w - margin] = wrist_small
    status = "SUCCESS" if success else "running"
    text = f"{label} | step={step} | {status}"
    bar_h = max(34, h // 12)
    font_scale = max(0.48, h / 512 * 0.65)
    thickness = max(1, h // 400)
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (margin, h - max(10, bar_h // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def make_env(task_suite: Any, task_id: int, task_suite_name: str, camera_size: int, seed: int) -> tuple[Any, str]:
    task = task_suite.get_task(task_id)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    env.seed(seed)
    return env, str(task.language)


def set_episode_state(env: Any, task_suite: Any, task_id: int, episode_index: int) -> dict[str, Any]:
    init_states = task_suite.get_task_init_states(task_id)
    if episode_index < 0 or episode_index >= len(init_states):
        raise IndexError(f"episode_index={episode_index} out of range for init_states length={len(init_states)}")
    env.reset()
    return env.set_init_state(init_states[episode_index])


@torch.no_grad()
def predict_action_chunk(bundle: PolicyBundle, skill_spec: Any, raw_obs: dict[str, Any], task: str) -> np.ndarray:
    raw_batch = raw_batch_from_obs(raw_obs, task)
    proc_batch = bundle.preprocessor(raw_batch)
    pred = bundle.model.predict_action_chunk(proc_batch)
    pred = pred[:, :, : skill_spec.action_dim]
    pred = bundle.postprocessor(pred)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)


def rollout_once(
    *,
    bundle: PolicyBundle,
    skill_spec: Any,
    task_suite: Any,
    task_suite_name: str,
    task_id: int,
    episode_index: int,
    seed: int,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    camera_size: int,
    progress_every: int,
) -> RolloutResult:
    env, task_description = make_env(task_suite, task_id, task_suite_name, camera_size, seed)
    try:
        raw_obs = set_episode_state(env, task_suite, task_id, episode_index)
        frames: list[np.ndarray] = []
        plan: deque[np.ndarray] = deque()
        success = False
        total_steps = 0

        for wait_step in range(num_steps_wait):
            raw_obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
            frames.append(render_frame(raw_obs, label=bundle.name, success=False, step=wait_step))

        for step in range(max_steps):
            if progress_every > 0 and step > 0 and step % progress_every == 0:
                print(
                    f"[rollout] episode_index={episode_index} {bundle.name} step={step}/{max_steps}",
                    flush=True,
                )
            if not plan:
                chunk = predict_action_chunk(bundle, skill_spec, raw_obs, task_description)
                if len(chunk) < replan_steps:
                    raise RuntimeError(f"Predicted chunk has only {len(chunk)} steps; replan_steps={replan_steps}")
                plan.extend(chunk[:replan_steps])

            action = np.clip(plan.popleft(), -1.0, 1.0)
            raw_obs, _, done, _ = env.step(action.tolist())
            success = bool(done or env.check_success())
            total_steps = step + 1
            frames.append(render_frame(raw_obs, label=bundle.name, success=success, step=total_steps))
            if success:
                break

        return RolloutResult(
            policy_name=bundle.name,
            success=success,
            steps=total_steps,
            episode_index=episode_index,
            seed=seed,
            frames=frames,
        )
    finally:
        env.close()


def write_video(
    path: Path,
    frames: list[np.ndarray],
    fps: int,
    *,
    codec: str = "libx264",
    crf: int = 16,
    preset: str = "slow",
) -> None:
    if not frames:
        raise ValueError("Cannot write empty video.")
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(stderr_text)
        return
    except Exception as exc:
        print(f"[video] ffmpeg writer failed for {path}: {exc}; trying imageio.", flush=True)

    try:
        import imageio.v2 as imageio

        output_params = ["-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
        with imageio.get_writer(
            str(path),
            fps=fps,
            codec=codec,
            macro_block_size=1,
            output_params=output_params,
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return
    except Exception as exc:
        print(f"[video] imageio/ffmpeg writer failed for {path}: {exc}; falling back to OpenCV mp4v.", flush=True)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_side_by_side(
    path: Path,
    left: RolloutResult,
    right: RolloutResult,
    fps: int,
    *,
    codec: str = "libx264",
    crf: int = 16,
    preset: str = "slow",
) -> None:
    n = max(len(left.frames), len(right.frames))
    left_frames = left.frames + [left.frames[-1]] * (n - len(left.frames))
    right_frames = right.frames + [right.frames[-1]] * (n - len(right.frames))
    frames = [np.concatenate([l, r], axis=1) for l, r in zip(left_frames, right_frames, strict=True)]
    write_video(path, frames, fps, codec=codec, crf=crf, preset=preset)


def load_policy_bundle(
    *,
    name: str,
    skill_spec: Any,
    stats: dict[str, Any],
    base_model_path: Path,
    adapter_dir: Path | None,
    device: str,
    dtype: str,
    tokenizer_name_or_path: str,
) -> PolicyBundle:
    preprocessor, postprocessor = build_processors(
        skill_spec,
        stats,
        device=device,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    if adapter_dir is None:
        model = load_skill_base_policy(skill_spec, base_model_path=base_model_path, device=device, dtype=dtype)
    else:
        model = load_skill_peft_policy(
            skill_spec,
            adapter_dir,
            base_model_path=base_model_path,
            device=device,
            dtype=dtype,
        )
    model.eval()
    return PolicyBundle(name=name, model=model, preprocessor=preprocessor, postprocessor=postprocessor, device=device)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_grad_enabled(False)

    skill_spec = load_skill_spec(args.skill_root, args.skill_id)
    stats = load_stats(skill_spec)
    adapter_dir = args.adapter_dir or args.output_root / args.skill_id / args.group / args.run_name / "best"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    max_steps = args.max_steps or LIBERO_SUITE_MAX_STEPS.get(args.task_suite, 300)
    task = task_suite.get_task(args.task_id)
    print(
        json.dumps(
            {
                "skill_id": args.skill_id,
                "task_suite": args.task_suite,
                "task_id": args.task_id,
                "task": task.language,
                "adapter_dir": str(adapter_dir),
                "max_steps": max_steps,
                "episodes": [args.start_episode, args.start_episode + args.num_episodes - 1],
            },
            indent=2,
        ),
        flush=True,
    )

    base_bundle = load_policy_bundle(
        name="PI05 base",
        skill_spec=skill_spec,
        stats=stats,
        base_model_path=args.base_model_path,
        adapter_dir=None,
        device=args.base_device,
        dtype=args.dtype,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
    )
    adapter_bundle = load_policy_bundle(
        name=f"PI05 LoRA {args.group}/{args.run_name}",
        skill_spec=skill_spec,
        stats=stats,
        base_model_path=args.base_model_path,
        adapter_dir=adapter_dir,
        device=args.adapter_device,
        dtype=args.dtype,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
    )

    manifest: list[dict[str, Any]] = []
    for episode_index in range(args.start_episode, args.start_episode + args.num_episodes):
        print(f"[rollout] episode_index={episode_index} base...", flush=True)
        base_result = rollout_once(
            bundle=base_bundle,
            skill_spec=skill_spec,
            task_suite=task_suite,
            task_suite_name=args.task_suite,
            task_id=args.task_id,
            episode_index=episode_index,
            seed=args.seed,
            max_steps=max_steps,
            num_steps_wait=args.num_steps_wait,
            replan_steps=args.replan_steps,
            camera_size=args.camera_size,
            progress_every=args.progress_every,
        )
        print(
            f"[rollout] episode_index={episode_index} base_success={base_result.success} "
            f"steps={base_result.steps}",
            flush=True,
        )

        print(f"[rollout] episode_index={episode_index} adapter...", flush=True)
        adapter_result = rollout_once(
            bundle=adapter_bundle,
            skill_spec=skill_spec,
            task_suite=task_suite,
            task_suite_name=args.task_suite,
            task_id=args.task_id,
            episode_index=episode_index,
            seed=args.seed,
            max_steps=max_steps,
            num_steps_wait=args.num_steps_wait,
            replan_steps=args.replan_steps,
            camera_size=args.camera_size,
            progress_every=args.progress_every,
        )
        print(
            f"[rollout] episode_index={episode_index} adapter_success={adapter_result.success} "
            f"steps={adapter_result.steps}",
            flush=True,
        )

        case_dir = args.out_dir / args.skill_id / f"{args.task_suite}_task{args.task_id:02d}_episode{episode_index:03d}"
        should_save = args.save_all_candidates or ((not base_result.success) and adapter_result.success)
        item = {
            "episode_index": episode_index,
            "seed": args.seed,
            "base_success": base_result.success,
            "base_steps": base_result.steps,
            "adapter_success": adapter_result.success,
            "adapter_steps": adapter_result.steps,
            "saved": should_save,
        }
        if should_save:
            base_video = case_dir / "base_failure.mp4"
            adapter_video = case_dir / "lora_success.mp4"
            side_by_side = case_dir / "base_vs_lora_side_by_side.mp4"
            write_video(
                base_video,
                base_result.frames,
                args.video_fps,
                codec=args.video_codec,
                crf=args.video_crf,
                preset=args.video_preset,
            )
            write_video(
                adapter_video,
                adapter_result.frames,
                args.video_fps,
                codec=args.video_codec,
                crf=args.video_crf,
                preset=args.video_preset,
            )
            write_side_by_side(
                side_by_side,
                base_result,
                adapter_result,
                args.video_fps,
                codec=args.video_codec,
                crf=args.video_crf,
                preset=args.video_preset,
            )
            item.update(
                {
                    "base_video": str(base_video),
                    "adapter_video": str(adapter_video),
                    "side_by_side_video": str(side_by_side),
                }
            )
        manifest.append(item)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"{args.skill_id}_rollout_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if (not base_result.success) and adapter_result.success and not args.run_all:
            print("[rollout] found target case: base failed, adapter succeeded.", flush=True)
            print(json.dumps(item, indent=2, ensure_ascii=False), flush=True)
            return

    if any((not item["base_success"]) and item["adapter_success"] for item in manifest):
        print("[rollout] completed requested range and found at least one base-fail/adapter-success case.", flush=True)
    else:
        print("[rollout] no base-fail/adapter-success case found in requested range.", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
