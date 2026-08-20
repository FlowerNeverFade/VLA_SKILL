#!/usr/bin/env python
"""HTTP policy server for PI05 base and many skill-LoRA adapters."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from collections import deque
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from vla_skill.constants import (
    DATA_ROOT,
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_LOCAL_TOKENIZER_PATH,
    DEFAULT_SKILL_ROOT,
    IMAGE_RESOLUTION,
)
from vla_skill.dataset import load_skill_spec, load_stats
from vla_skill.lora import build_lora_config
from vla_skill.pi05 import build_processors, load_base_policy, make_skill_policy_config


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
    target_h, target_w = IMAGE_RESOLUTION
    if rgb.shape[:2] != (target_h, target_w):
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return tensor


def _set_adapter(model: Any, adapter_name: str) -> None:
    try:
        model.set_adapter(adapter_name, inference_mode=True)
    except TypeError:
        model.set_adapter(adapter_name)


class MultiSkillPi05Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        self.adapter_dirs = {
            item["skill_id"]: Path(item["adapter_dir"])
            for item in self.manifest.get("tasks", [])
        }
        if not self.adapter_dirs:
            raise ValueError(f"No tasks found in manifest: {args.manifest}")

        self.spec_cache: dict[str, Any] = {}
        self.processor_cache: dict[str, tuple[Any, Any]] = {}
        self.loaded_adapters: set[str] = set()
        self.queues: dict[tuple[str, str], deque[np.ndarray]] = {}

        first_skill_id = next(iter(self.adapter_dirs))
        first_spec = self._get_skill_spec(first_skill_id)
        config = make_skill_policy_config(
            first_spec,
            base_model_path=args.base_model_path,
            device=args.device,
            dtype=args.dtype,
            train_expert_only=True,
            gradient_checkpointing=False,
            compile_model=False,
        )
        base_policy = load_base_policy(config, base_model_path=args.base_model_path, strict=True)
        self.model = base_policy.wrap_with_peft(
            peft_config=build_lora_config(
                args.lora_group,
                base_model_name_or_path=str(args.base_model_path),
                inference_mode=True,
            )
        )
        self.model.eval()
        self.last_status: dict[str, Any] = {
            "ready": True,
            "base_model_path": str(args.base_model_path),
            "device": args.device,
            "dtype": args.dtype,
            "manifest": str(args.manifest),
            "skill_count": len(self.adapter_dirs),
            "loaded_adapter_count": 0,
        }

    def _get_skill_spec(self, skill_id: str):
        cached = self.spec_cache.get(skill_id)
        if cached is None:
            cached = load_skill_spec(self.args.skill_root, skill_id)
            self.spec_cache[skill_id] = cached
        return cached

    def _get_processors(self, skill_id: str):
        cached = self.processor_cache.get(skill_id)
        if cached is None:
            spec = self._get_skill_spec(skill_id)
            cached = build_processors(
                spec,
                load_stats(spec),
                device=self.args.device,
                tokenizer_name_or_path=self.args.tokenizer_name_or_path,
            )
            self.processor_cache[skill_id] = cached
        return cached

    def _ensure_adapter(self, skill_id: str) -> str:
        adapter_dir = self.adapter_dirs.get(skill_id)
        if adapter_dir is None:
            raise KeyError(f"skill_id {skill_id!r} is not in manifest")
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"missing adapter_model.safetensors under {adapter_dir}")

        adapter_name = f"skill_lora__{skill_id}"
        if adapter_name not in self.loaded_adapters:
            self.model.load_adapter(str(adapter_dir), adapter_name=adapter_name, is_trainable=False)
            self.loaded_adapters.add(adapter_name)
        return adapter_name

    def reset(self, mode: str | None = None, skill_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            if mode and skill_id:
                self.queues.pop((mode, skill_id), None)
            elif skill_id:
                for key in list(self.queues):
                    if key[1] == skill_id:
                        self.queues.pop(key, None)
            else:
                self.queues.clear()
            if hasattr(self.model, "reset"):
                self.model.reset()
            self.last_status = {**self.last_status, "last_reset": {"mode": mode, "skill_id": skill_id}}
            return {"ok": True, "reset": True, "mode": mode, "skill_id": skill_id}

    def _build_raw_batch(self, payload: dict[str, Any], skill_id: str) -> dict[str, Any]:
        spec = self._get_skill_spec(skill_id)
        state = np.asarray(payload.get("state", []), dtype=np.float32)
        if state.shape != (spec.state_dim,):
            raise ValueError(f"expected state shape {(spec.state_dim,)}, got {state.shape}")

        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("payload.images must be an object")

        raw_batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": str(payload.get("task") or spec.display_name),
        }
        for image_name, feature_name in IMAGE_KEYS.items():
            if image_name not in images:
                raise ValueError(f"missing image {image_name}")
            raw_batch[feature_name] = _decode_image(images[image_name])
        return raw_batch

    def _predict_chunk(self, payload: dict[str, Any], mode: str, skill_id: str) -> np.ndarray:
        spec = self._get_skill_spec(skill_id)
        preprocessor, postprocessor = self._get_processors(skill_id)
        self.model.config.chunk_size = spec.chunk_size
        self.model.config.n_action_steps = spec.n_action_steps
        self.model.config.output_features["action"].shape = (spec.action_dim,)

        if mode == "lora":
            adapter_name = self._ensure_adapter(skill_id)
            _set_adapter(self.model, adapter_name)
            adapter_context = nullcontext()
        elif mode == "base":
            adapter_name = None
            adapter_context = self.model.disable_adapter()
        else:
            raise ValueError(f"unsupported mode {mode!r}; expected 'base' or 'lora'")

        raw_batch = self._build_raw_batch(payload, skill_id)
        proc_batch = preprocessor(raw_batch)
        with torch.inference_mode(), adapter_context:
            pred = self.model.predict_action_chunk(proc_batch)
            pred = pred[:, :, : spec.action_dim]
            pred = postprocessor(pred)
        actions = np.asarray(pred.squeeze(0).detach().cpu(), dtype=np.float32)
        self.last_status = {
            **self.last_status,
            "last_mode": mode,
            "last_skill_id": skill_id,
            "last_adapter_name": adapter_name,
            "loaded_adapter_count": len(self.loaded_adapters),
            "action_shape": list(actions.shape),
        }
        return actions

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "lora").lower()
        skill_id = str(payload.get("skill_id") or "")
        if not skill_id:
            raise ValueError("payload.skill_id is required")

        with self.lock:
            key = (mode, skill_id)
            queue = self.queues.setdefault(key, deque())
            if not queue:
                actions = self._predict_chunk(payload, mode, skill_id)
                for action in actions:
                    queue.append(action.copy())
            action = queue.popleft()
            response = {
                "ok": True,
                "mode": mode,
                "skill_id": skill_id,
                "action": action.tolist(),
                "remaining_chunk": len(queue),
                "loaded_adapter_count": len(self.loaded_adapters),
            }
            self.last_status = {**self.last_status, **{k: v for k, v in response.items() if k != "action"}}
            return response


class PolicyHandler(BaseHTTPRequestHandler):
    runtime: MultiSkillPi05Runtime

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, self.runtime.last_status)
            return
        if parsed.path == "/reset":
            query = parse_qs(parsed.query)
            mode = query.get("mode", [None])[0]
            skill_id = query.get("skill_id", [None])[0]
            self._send_json(200, self.runtime.reset(mode=mode, skill_id=skill_id))
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            if self.path == "/act":
                response = self.runtime.act(payload)
            elif self.path == "/reset":
                response = self.runtime.reset(
                    mode=payload.get("mode"),
                    skill_id=payload.get("skill_id"),
                )
            else:
                self.send_error(404)
                return
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6020)
    parser.add_argument("--manifest", type=Path, default=CODE_ROOT / "robocasa_pickplace_50_skill_lora_manifest.json")
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--tokenizer-name-or-path", type=Path, default=DEFAULT_LOCAL_TOKENIZER_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lora-group", default="C")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)
    PolicyHandler.runtime = MultiSkillPi05Runtime(args)
    with ThreadingHTTPServer((args.host, args.port), PolicyHandler) as server:
        print(f"PI05 multi-skill policy server ready: http://{args.host}:{args.port}", flush=True)
        print(json.dumps(PolicyHandler.runtime.last_status, indent=2, ensure_ascii=False), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
