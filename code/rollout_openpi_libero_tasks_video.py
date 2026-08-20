#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy

try:
    import torch

    _ORIGINAL_TORCH_LOAD = torch.load
    if "weights_only" in inspect.signature(torch.load).parameters:

        def _torch_load_libero_compat(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return _ORIGINAL_TORCH_LOAD(*args, **kwargs)

        torch.load = _torch_load_libero_compat
except Exception:
    torch = None  # type: ignore[assignment]

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out an OpenPI policy server on many LIBERO tasks.")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-name", default="pi05_libero")
    parser.add_argument("--task-suite", default="libero_90")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--camera-size", type=int, default=768)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=14)
    parser.add_argument("--video-preset", default="slow")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "rollout_videos_libero90_50tasks",
    )
    return parser.parse_args()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
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


def policy_observation(raw_obs: dict[str, Any], task_description: str, resize_size: int) -> dict[str, Any]:
    image = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": libero_state(raw_obs),
        "prompt": str(task_description),
    }


def draw_label(frame: np.ndarray, *, label: str, task_text: str, step: int, success: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = max(72, h // 8)
    margin = max(10, w // 70)
    font_scale = max(0.58, h / 768 * 0.72)
    small_scale = max(0.46, h / 768 * 0.54)
    thickness = max(1, h // 360)
    output = frame.copy()
    cv2.rectangle(output, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    status = "SUCCESS" if success else "running"
    cv2.putText(
        output,
        f"{label} | step={step} | {status}",
        (margin, h - bar_h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    max_chars = max(42, int(w / 12))
    clipped = task_text if len(task_text) <= max_chars else task_text[: max_chars - 3] + "..."
    cv2.putText(
        output,
        clipped,
        (margin, h - max(18, bar_h // 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        (220, 220, 220),
        thickness,
        cv2.LINE_AA,
    )
    return output


def render_frame(raw_obs: dict[str, Any], *, label: str, task_text: str, step: int, success: bool) -> np.ndarray:
    base_img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]).copy()
    wrist_img = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]).copy()
    h, w = base_img.shape[:2]
    inset_w = max(160, w // 3)
    inset_h = max(160, h // 3)
    margin = max(10, w // 64)
    wrist_small = cv2.resize(wrist_img, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    frame = base_img
    frame[margin : margin + inset_h, w - inset_w - margin : w - margin] = wrist_small
    return draw_label(frame, label=label, task_text=task_text, step=step, success=success)


class FfmpegVideoWriter:
    def __init__(self, path: Path, *, fps: int, codec: str, crf: int, preset: str) -> None:
        self.path = path
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.proc: subprocess.Popen[bytes] | None = None
        self.frame_count = 0

    def _start(self, frame: np.ndarray) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
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
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            self.codec,
            "-crf",
            str(self.crf),
            "-preset",
            self.preset,
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None:
            self._start(frame)
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self.proc is None:
            return
        assert self.proc.stdin is not None
        stdout, stderr = self.proc.communicate()
        if self.proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg failed for {self.path}: {stderr_text}")


def make_env(task_suite: Any, task_id: int, camera_size: int, seed: int) -> tuple[Any, str]:
    task = task_suite.get_task(task_id)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    env.seed(seed)
    return env, str(task.language)


def set_episode_state(env: Any, task_suite: Any, task_id: int, init_state_id: int) -> dict[str, Any]:
    init_states = task_suite.get_task_init_states(task_id)
    if init_state_id < 0 or init_state_id >= len(init_states):
        raise IndexError(f"init_state_id={init_state_id} out of range for task_id={task_id}: {len(init_states)} states")
    env.reset()
    return env.set_init_state(init_states[init_state_id])


def rollout_task(
    *,
    client: Any,
    task_suite: Any,
    task_suite_name: str,
    task_id: int,
    init_state_id: int,
    seed: int,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    camera_size: int,
    resize_size: int,
    policy_name: str,
    video_path: Path,
    video_fps: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
    progress_every: int,
) -> dict[str, Any]:
    env, task_description = make_env(task_suite, task_id, camera_size, seed)
    writer = FfmpegVideoWriter(
        video_path,
        fps=video_fps,
        codec=video_codec,
        crf=video_crf,
        preset=video_preset,
    )
    try:
        raw_obs = set_episode_state(env, task_suite, task_id, init_state_id)
        plan: deque[np.ndarray] = deque()
        success = False
        total_steps = 0

        label = f"{policy_name} | {task_suite_name} task={task_id:03d}"
        for wait_step in range(num_steps_wait):
            raw_obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            writer.write(render_frame(raw_obs, label=label, task_text=task_description, step=wait_step, success=False))

        for step in range(max_steps):
            if progress_every > 0 and step > 0 and step % progress_every == 0:
                print(f"[rollout] {task_suite_name} task={task_id:03d} step={step}/{max_steps}", flush=True)
            if not plan:
                element = policy_observation(raw_obs, task_description, resize_size)
                action_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)
                if len(action_chunk) < replan_steps:
                    raise RuntimeError(f"action chunk has {len(action_chunk)} steps, replan_steps={replan_steps}")
                plan.extend(action_chunk[:replan_steps])

            action = np.clip(np.asarray(plan.popleft(), dtype=np.float32), -1.0, 1.0)
            raw_obs, _, done, _ = env.step(action.tolist())
            success = bool(done or env.check_success())
            total_steps = step + 1
            writer.write(render_frame(raw_obs, label=label, task_text=task_description, step=total_steps, success=success))
            if success:
                break

        writer.close()
        return {
            "task_suite": task_suite_name,
            "task_id": int(task_id),
            "init_state_id": int(init_state_id),
            "task": task_description,
            "success": bool(success),
            "steps": int(total_steps),
            "video_path": str(video_path),
            "video_frames": int(writer.frame_count),
        }
    finally:
        try:
            writer.close()
        finally:
            env.close()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if args.task_ids:
        task_ids = parse_int_list(args.task_ids)
    else:
        task_ids = list(range(args.start_task, min(args.start_task + args.num_tasks, task_suite.n_tasks)))
    max_steps = args.max_steps or MAX_STEPS_BY_SUITE.get(args.task_suite, 300)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"{args.policy_name}_{args.task_suite}_manifest.json"

    print(
        json.dumps(
            {
                "policy": args.policy_name,
                "policy_server": f"{args.policy_host}:{args.policy_port}",
                "task_suite": args.task_suite,
                "task_ids": task_ids,
                "init_state_id": args.init_state_id,
                "max_steps": max_steps,
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
    manifest: list[dict[str, Any]] = []
    for index, task_id in enumerate(task_ids, start=1):
        task_dir = args.out_dir / args.task_suite / f"task_{task_id:03d}"
        video_path = task_dir / f"{args.policy_name}_{args.task_suite}_task{task_id:03d}_init{args.init_state_id:03d}.mp4"
        print(f"[rollout] ({index}/{len(task_ids)}) task_id={task_id} start", flush=True)
        try:
            item = rollout_task(
                client=client,
                task_suite=task_suite,
                task_suite_name=args.task_suite,
                task_id=task_id,
                init_state_id=args.init_state_id,
                seed=args.seed,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                replan_steps=args.replan_steps,
                camera_size=args.camera_size,
                resize_size=args.resize_size,
                policy_name=args.policy_name,
                video_path=video_path,
                video_fps=args.video_fps,
                video_codec=args.video_codec,
                video_crf=args.video_crf,
                video_preset=args.video_preset,
                progress_every=args.progress_every,
            )
            item["ok"] = True
            print(
                f"[rollout] ({index}/{len(task_ids)}) task_id={task_id} "
                f"success={item['success']} steps={item['steps']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            item = {
                "ok": False,
                "task_suite": args.task_suite,
                "task_id": int(task_id),
                "init_state_id": int(args.init_state_id),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[rollout] ({index}/{len(task_ids)}) task_id={task_id} ERROR {item['error']}", flush=True)
        manifest.append(item)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    successes = sum(bool(item.get("success")) for item in manifest)
    print(
        json.dumps(
            {
                "completed": len(manifest),
                "successes": successes,
                "success_rate": successes / len(manifest) if manifest else 0.0,
                "manifest": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
