#!/usr/bin/env python
"""HTTP policy server for running a local LeRobot PI05 checkpoint."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parent / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from vla_skill.constants import DEFAULT_BASE_MODEL_PATH, DEFAULT_LOCAL_TOKENIZER_PATH, DEFAULT_SKILL_ROOT
from vla_skill.dataset import load_skill_spec, load_stats
from vla_skill.pi05 import build_processors, load_skill_base_policy, load_skill_peft_policy


IMAGE_KEYS = {
    "base_0_rgb": "observation.images.base_0_rgb",
    "left_wrist_0_rgb": "observation.images.left_wrist_0_rgb",
    "right_wrist_0_rgb": "observation.images.right_wrist_0_rgb",
}


def _decode_image(payload: str) -> torch.Tensor:
    raw = base64.b64decode(payload)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor


class Pi05Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.skill_spec = load_skill_spec(args.skill_root, args.skill_id)
        if args.adapter_dir:
            self.model = load_skill_peft_policy(
                self.skill_spec,
                args.adapter_dir,
                base_model_path=args.base_model_path,
                device=args.device,
                dtype=args.dtype,
            )
            self.model_type = "pi05+lora"
        else:
            self.model = load_skill_base_policy(
                self.skill_spec,
                base_model_path=args.base_model_path,
                device=args.device,
                dtype=args.dtype,
            )
            self.model_type = "pi05_base"
        self.preprocessor, self.postprocessor = build_processors(
            self.skill_spec,
            load_stats(self.skill_spec),
            device=args.device,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
        )
        self.queue: deque[np.ndarray] = deque()
        self.last_status = {
            "ready": True,
            "skill_id": self.skill_spec.skill_id,
            "task": self.skill_spec.display_name,
            "device": args.device,
            "dtype": args.dtype,
            "model_type": self.model_type,
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "action_dim": self.skill_spec.action_dim,
            "state_dim": self.skill_spec.state_dim,
        }

    def reset(self) -> None:
        with self.lock:
            self.queue.clear()
            if hasattr(self.model, "reset"):
                self.model.reset()

    def _build_raw_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(payload.get("state", []), dtype=np.float32)
        if state.shape != (self.skill_spec.state_dim,):
            raise ValueError(f"expected state shape {(self.skill_spec.state_dim,)}, got {state.shape}")

        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("payload.images must be an object")

        raw_batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": str(payload.get("task") or self.skill_spec.display_name),
        }
        for image_name, feature_name in IMAGE_KEYS.items():
            if image_name not in images:
                raise ValueError(f"missing image {image_name}")
            raw_batch[feature_name] = _decode_image(images[image_name])
        return raw_batch

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.queue:
                raw_batch = self._build_raw_batch(payload)
                proc_batch = self.preprocessor(raw_batch)
                with torch.inference_mode():
                    pred = self.model.predict_action_chunk(proc_batch)
                    pred = pred[:, :, : self.skill_spec.action_dim]
                    pred = self.postprocessor(pred)
                actions = np.asarray(pred.squeeze(0).detach().cpu(), dtype=np.float32)
                for action in actions:
                    self.queue.append(action.copy())

            action = self.queue.popleft()
            response = {
                "action": action.tolist(),
                "skill_id": self.skill_spec.skill_id,
                "remaining_chunk": len(self.queue),
            }
            self.last_status = {**self.last_status, **response, "ready": True}
            return response


class PolicyHandler(BaseHTTPRequestHandler):
    runtime: Pi05Runtime

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, self.runtime.last_status)
            return
        if self.path == "/reset":
            self.runtime.reset()
            self._send_json(200, {"ok": True, "reset": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/act":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            response = self.runtime.act(payload)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, {"ok": True, **response})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6010)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--skill-id", default="robocasa_target_atomic_024_open_cabinet_doors")
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--tokenizer-name-or-path", type=Path, default=DEFAULT_LOCAL_TOKENIZER_PATH)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    PolicyHandler.runtime = Pi05Runtime(args)
    with ThreadingHTTPServer((args.host, args.port), PolicyHandler) as server:
        print(f"PI05 policy server ready: http://{args.host}:{args.port}")
        print(json.dumps(PolicyHandler.runtime.last_status, indent=2))
        server.serve_forever()


if __name__ == "__main__":
    main()
