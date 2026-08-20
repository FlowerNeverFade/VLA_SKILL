#!/usr/bin/env python
"""Generate RoboCasa base-vs-skill-LoRA comparison videos."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from vla_skill.constants import DATA_ROOT

ROBOCASA_ROOT = DATA_ROOT / "datasets" / "RoboCasa"
ROBOSUITE_ROOT = DATA_ROOT / "datasets" / "robosuite"
for path in (str(ROBOSUITE_ROOT), str(ROBOCASA_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import numpy as np
import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config


CAMERA_MAP = (
    ("base_0_rgb", "robot0_agentview_left_image", "agentview_left"),
    ("left_wrist_0_rgb", "robot0_eye_in_hand_image", "eye_in_hand"),
    ("right_wrist_0_rgb", "robot0_agentview_right_image", "agentview_right"),
)


def _unwrap_reset(reset_out):
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out
    return reset_out, {}


def _unwrap_step(step_out):
    if len(step_out) == 5:
        return step_out
    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, done, False, info
    raise RuntimeError(f"unexpected step() return size: {len(step_out)}")


def _robocasa_state16(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    # Matches RoboCasa LeRobot `observation.state`:
    # base pose, end-effector pose in the mobile-base frame, and gripper qpos.
    parts = [
        obs.get("robot0_base_pos", np.zeros(3, dtype=np.float32)),
        obs.get("robot0_base_quat", np.array([0, 0, 0, 1], dtype=np.float32)),
        obs.get("robot0_base_to_eef_pos", np.zeros(3, dtype=np.float32)),
        obs.get("robot0_base_to_eef_quat", np.array([0, 0, 0, 1], dtype=np.float32)),
        obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32)),
    ]
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts])[:16]


def _encode_jpeg_rgb(image: np.ndarray, size: int, quality: int) -> str:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode()


def _policy_payload(obs: Mapping[str, np.ndarray], task: dict[str, Any], mode: str, args: argparse.Namespace) -> dict[str, Any]:
    images = {}
    for target_name, obs_key, _label in CAMERA_MAP:
        image = obs.get(obs_key)
        if image is None:
            raise RuntimeError(f"missing observation image {obs_key}")
        images[target_name] = _encode_jpeg_rgb(image, args.policy_image_size, args.policy_jpeg_quality)
    return {
        "mode": mode,
        "skill_id": task["skill_id"],
        "task": task["task_name"],
        "state": _robocasa_state16(obs).tolist(),
        "images": images,
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


class FfmpegVideoWriter:
    def __init__(self, path: Path, *, fps: int, width: int, height: int, codec: str, crf: int, preset: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = width
        self.height = height
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"expected frame shape {(self.height, self.width)}, got {frame.shape[:2]}")
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        stderr = self.proc.stderr.read() if self.proc.stderr is not None else b""
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {stderr.decode(errors='replace')[-2000:]}")


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _make_panel(
    obs: Mapping[str, np.ndarray],
    *,
    task: dict[str, Any],
    mode: str,
    step_idx: int,
    reward: float,
    success: bool,
    first_success_step: int | None,
    action: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    panels = []
    for _target_name, obs_key, label in CAMERA_MAP:
        frame = obs.get(obs_key)
        if frame is None:
            frame = np.zeros((args.panel_height, args.panel_width, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (args.panel_width, args.panel_height), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame, dtype=np.uint8).copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (args.panel_width, 34), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        _draw_text(frame, label, (10, 23), 0.62, (255, 255, 255), 2)
        panels.append(frame)

    panel = np.concatenate(panels, axis=1)
    header_h = 86
    out = np.zeros((panel.shape[0] + header_h, panel.shape[1], 3), dtype=np.uint8)
    out[header_h:] = panel

    if mode == "base":
        title = "PI0.5 BASE"
        color = (105, 190, 255)
    else:
        title = "PI0.5 + SKILL-LoRA"
        color = (120, 245, 160)
    status = "success" if success else "running"
    success_text = "none" if first_success_step is None else str(first_success_step)
    action_norm = float(np.linalg.norm(action)) if action.size else 0.0
    line1 = f"{title} | task {task['rank']:02d}/50 | {task['obj_group']} -> cabinet"
    line2 = f"step={step_idx:03d} reward={reward:.3f} status={status} first_success={success_text} action_norm={action_norm:.3f}"
    line3 = task["task_name"][:150]
    _draw_text(out, line1, (14, 28), 0.78, color, 2)
    _draw_text(out, line2, (14, 54), 0.58, (235, 235, 235), 1)
    _draw_text(out, line3, (14, 78), 0.52, (220, 220, 220), 1)
    return out


def _make_env(task: dict[str, Any], args: argparse.Namespace, seed: int):
    controller = load_composite_controller_config(controller=None, robot=args.robot)
    return robosuite.make(
        env_name=task.get("env_name") or args.env_name,
        robots=args.robot,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_gpu_device_id=args.render_gpu,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"],
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        camera_depths=False,
        control_freq=args.control_freq,
        seed=seed,
        layout_ids=[args.layout_id],
        style_ids=[args.style_id],
        obj_registries=("objaverse", "aigen", "lightwheel"),
        **(task.get("env_kwargs") or {"obj_groups": task["obj_group"]}),
    )


def _reset_policy(policy_url: str, mode: str, skill_id: str, timeout: float) -> None:
    query = urlencode({"mode": mode, "skill_id": skill_id})
    _get_json(f"{policy_url.rstrip('/')}/reset?{query}", timeout)


def _clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray, scale: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < low.shape[0]:
        action = np.pad(action, (0, low.shape[0] - action.shape[0]))
    elif action.shape[0] > low.shape[0]:
        action = action[: low.shape[0]]
    return np.clip(action * scale, low, high).astype(np.float32)


def run_rollout(task: dict[str, Any], mode: str, out_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    seed = args.seed + int(task["rank"]) * args.seed_stride
    env = _make_env(task, args, seed)
    writer: FfmpegVideoWriter | None = None
    first_success_step: int | None = None
    last_success = False
    last_reward = 0.0
    start = time.time()
    try:
        obs, _ = _unwrap_reset(env.reset())
        low, high = env.action_spec
        _reset_policy(args.policy_url, mode, task["skill_id"], args.policy_timeout)
        panel_w = args.panel_width * len(CAMERA_MAP)
        panel_h = args.panel_height + 86
        writer = FfmpegVideoWriter(
            out_path,
            fps=args.video_fps,
            width=panel_w,
            height=panel_h,
            codec=args.video_codec,
            crf=args.video_crf,
            preset=args.video_preset,
        )

        for step_idx in range(args.max_steps):
            payload = _policy_payload(obs, task, mode, args)
            response = _post_json(f"{args.policy_url.rstrip('/')}/act", payload, args.policy_timeout)
            if not response.get("ok"):
                raise RuntimeError(str(response))
            action = _clip_action(np.asarray(response["action"], dtype=np.float32), low, high, args.policy_scale)
            obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))
            last_reward = float(reward)
            success = bool(env._check_success()) if hasattr(env, "_check_success") else bool(info.get("success", False))
            last_success = success
            if success and first_success_step is None:
                first_success_step = step_idx

            frame = _make_panel(
                obs,
                task=task,
                mode=mode,
                step_idx=step_idx,
                reward=last_reward,
                success=success,
                first_success_step=first_success_step,
                action=action,
                args=args,
            )
            writer.write(frame)
            if args.progress_every and (step_idx + 1) % args.progress_every == 0:
                print(
                    f"[rollout] task={task['rank']:02d} mode={mode} step={step_idx + 1}/{args.max_steps} "
                    f"success={success} first_success={first_success_step}",
                    flush=True,
                )
            if terminated and not args.ignore_done:
                break
            if truncated and not args.ignore_done:
                break
    finally:
        if writer is not None:
            writer.close()
        env.close()
    return {
        "mode": mode,
        "seed": seed,
        "success": last_success,
        "first_success_step": first_success_step,
        "last_reward": last_reward,
        "video": str(out_path),
        "elapsed_sec": round(time.time() - start, 3),
    }


def combine_videos(base_video: Path, lora_video: Path, out_path: Path, args: argparse.Namespace) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(base_video),
        "-i",
        str(lora_video),
        "-filter_complex",
        "[0:v][1:v]vstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-vcodec",
        args.video_codec,
        "-preset",
        args.video_preset,
        "-crf",
        str(args.video_crf),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "robocasa_pickplace_50_skill_lora_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "outputs" / "rollout_videos_robocasa_skill_lora_50_hq")
    parser.add_argument("--policy-url", default="http://127.0.0.1:6020")
    parser.add_argument("--env-name", default="PickPlaceCounterToCabinet")
    parser.add_argument("--robot", default="PandaOmron")
    parser.add_argument("--layout-id", type=int, default=39)
    parser.add_argument("--style-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seed-stride", type=int, default=17)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=480)
    parser.add_argument("--policy-image-size", type=int, default=224)
    parser.add_argument("--policy-jpeg-quality", type=int, default=95)
    parser.add_argument("--policy-timeout", type=float, default=180.0)
    parser.add_argument("--policy-scale", type=float, default=1.0)
    parser.add_argument("--render-gpu", type=int, default=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")))
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=14)
    parser.add_argument("--video-preset", default="medium")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ignore-done", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    selected = [
        task for task in tasks
        if args.start_rank <= int(task["rank"]) < args.start_rank + args.num_tasks
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_path = args.out_dir / "robocasa_50_base_vs_skill_lora_manifest.json"

    health = _get_json(f"{args.policy_url.rstrip('/')}/health", args.policy_timeout)
    print(json.dumps({"policy_health": health, "selected_tasks": len(selected), "out_dir": str(args.out_dir)}, indent=2), flush=True)

    results: list[dict[str, Any]] = []
    if out_manifest_path.is_file():
        try:
            results = json.loads(out_manifest_path.read_text(encoding="utf-8")).get("results", [])
        except Exception:
            results = []
    done_skill_ids = {item.get("skill_id") for item in results if item.get("ok") and item.get("side_by_side_video")}

    for task in selected:
        task_dir = args.out_dir / f"{int(task['rank']):02d}_{task['skill_id']}"
        base_video = task_dir / "base.mp4"
        lora_video = task_dir / "skill_lora.mp4"
        side_video = task_dir / "base_vs_skill_lora.mp4"
        if args.skip_existing and task["skill_id"] in done_skill_ids and side_video.is_file():
            print(f"[skip] task={task['rank']:02d} {task['skill_id']}", flush=True)
            continue

        print(f"[task] {task['rank']:02d}/{len(tasks)} {task['skill_id']} obj={task['obj_group']}", flush=True)
        item: dict[str, Any] = {
            "rank": task["rank"],
            "task_index": task["task_index"],
            "skill_id": task["skill_id"],
            "obj_group": task["obj_group"],
            "task_name": task["task_name"],
            "adapter_dir": task["adapter_dir"],
            "ok": False,
        }
        try:
            base_result = run_rollout(task, "base", base_video, args)
            lora_result = run_rollout(task, "lora", lora_video, args)
            combine_videos(base_video, lora_video, side_video, args)
            item.update(
                {
                    "ok": True,
                    "base": base_result,
                    "lora": lora_result,
                    "side_by_side_video": str(side_video),
                }
            )
            print(
                f"[done] task={task['rank']:02d} base_success={base_result['success']} "
                f"lora_success={lora_result['success']} video={side_video}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            item.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] task={task['rank']:02d} {item['error']}", flush=True)
        results = [old for old in results if old.get("skill_id") != task["skill_id"]]
        results.append(item)
        out_manifest_path.write_text(
            json.dumps(
                {
                    "source_manifest": str(args.manifest),
                    "out_dir": str(args.out_dir),
                    "video": {
                        "panel_width": args.panel_width,
                        "panel_height": args.panel_height,
                        "camera_width": args.camera_width,
                        "camera_height": args.camera_height,
                        "fps": args.video_fps,
                        "codec": args.video_codec,
                        "crf": args.video_crf,
                        "preset": args.video_preset,
                    },
                    "results": sorted(results, key=lambda item: int(item.get("rank", 10**9))),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    ok_count = sum(1 for item in results if item.get("ok"))
    print(f"[summary] ok={ok_count}/{len(selected)} manifest={out_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
