#!/usr/bin/env python
"""Robustly download a LeRobot/Hugging Face dataset with resume and retries."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, snapshot_download


def should_auth(endpoint: str) -> bool:
    return "huggingface.co" in endpoint


def local_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
        and path.name not in {"download.log", "download_manifest.json"}
    ]


def load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def image_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return sorted(key for key in features if key.startswith("observation.images."))


def write_manifest(
    root: Path,
    repo_id: str,
    endpoint: str,
    sha: str,
    sibling_count: int,
    info: dict[str, Any],
) -> Path:
    files = local_files(root)
    manifest = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "endpoint": endpoint,
        "commit_sha": sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "local_dir": str(root),
        "sibling_count": sibling_count,
        "local_file_count": len(files),
        "local_size_bytes": sum(path.stat().st_size for path in files),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "image_keys": image_keys(info),
    }
    path = root / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def download_once(
    repo_id: str,
    local_dir: Path,
    endpoint: str,
    max_workers: int,
    etag_timeout: int,
) -> tuple[str, int, dict[str, Any]]:
    token = get_token() if should_auth(endpoint) else None
    api = HfApi(endpoint=endpoint, token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(
        f"[repo] repo={repo_id} endpoint={endpoint} sha={info.sha} siblings={len(info.siblings)}",
        flush=True,
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        max_workers=max_workers,
        etag_timeout=etag_timeout,
        token=token,
        endpoint=endpoint,
    )
    dataset_info = load_info(local_dir)
    return str(info.sha), len(info.siblings), dataset_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--etag-timeout", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=int, default=30)
    parser.add_argument("--retries", type=int, default=0, help="0 means retry forever")
    parser.add_argument("--expected-image-keys", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.local_dir)
    root.mkdir(parents=True, exist_ok=True)

    attempt = 0
    while args.retries == 0 or attempt < args.retries:
        attempt += 1
        try:
            print(f"[download] attempt={attempt}", flush=True)
            sha, sibling_count, info = download_once(
                repo_id=args.repo_id,
                local_dir=root,
                endpoint=args.endpoint,
                max_workers=args.max_workers,
                etag_timeout=args.etag_timeout,
            )
            keys = image_keys(info)
            if args.expected_image_keys and len(keys) != args.expected_image_keys:
                raise ValueError(
                    f"Expected {args.expected_image_keys} image keys, found {len(keys)}: {keys}"
                )
            manifest_path = write_manifest(
                root=root,
                repo_id=args.repo_id,
                endpoint=args.endpoint,
                sha=sha,
                sibling_count=sibling_count,
                info=info,
            )
            print(
                f"[done] manifest={manifest_path} episodes={info.get('total_episodes')} "
                f"frames={info.get('total_frames')} image_keys={keys}",
                flush=True,
            )
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[error] attempt={attempt} type={type(exc).__name__} detail={exc}", flush=True)
            time.sleep(args.sleep_seconds)

    raise RuntimeError(f"Failed to download {args.repo_id} after {args.retries} attempts")


if __name__ == "__main__":
    main()
