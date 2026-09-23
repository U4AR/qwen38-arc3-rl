"""Per-request trajectory log for reinforcement learning.

Enabled with ``RL_TRAJECTORY_LOG=1``. Every chat request the analyzer makes is
appended to ``<artifacts>/<run_stem>_rl.jsonl.gz`` with the exact prompt token
ids the server saw, the sampled completion token ids and their logprobs, and
the game level that was being played when the request was issued. The trainer
uses the level tag to credit each request to the level it was working on.

Images (a board image on every turn) are stored once per episode in
``<run_stem>_images.jsonl.gz`` and referenced from records by sha1.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, load_runtime_state


def rl_trajectory_log_enabled() -> bool:
    return os.environ.get("RL_TRAJECTORY_LOG", "").strip().lower() in {"1", "true", "yes", "on"}


def _image_urls(messages: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if isinstance(url, str):
                    urls.append(url)
    return urls


class RLTrajectoryLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.images_path = path.with_name(path.name.replace("_rl.jsonl.gz", "_images.jsonl.gz"))
        self._saved_images: set[str] = set()
        self._index = 0
        self._lock = threading.Lock()

    @classmethod
    def for_state_path(cls, state_path: Path) -> "RLTrajectoryLogger":
        name = state_path.name
        stem = name[: -len(RUNTIME_STATE_FILENAME)].rstrip("_") if name.endswith(RUNTIME_STATE_FILENAME) else state_path.stem
        return cls(state_path.parent / f"{stem}_rl.jsonl.gz")

    def record(
        self,
        *,
        state_path: Path,
        messages: list[dict[str, Any]] | None,
        result: Any,
        analysis_step: int | None,
        action_num: int | None,
    ) -> None:
        if not result.token_ids or not result.prompt_token_ids:
            return
        frame, _ = load_runtime_state(state_path)
        record = {
            "i": self._index,
            "t": time.time(),
            "analysis_step": analysis_step,
            "action_num": action_num,
            "level": frame.level if frame is not None else None,
            "step": frame.step if frame is not None else None,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
            "prompt_token_ids": result.prompt_token_ids,
            "token_ids": result.token_ids,
            "logprobs": result.logprobs,
        }
        record["video_replays"] = sum(
            1
            for m in messages or []
            if isinstance(m.get("content"), list)
            and any(isinstance(p, dict) and p.get("text") == "Requested video replay:" for p in m["content"])
        )
        urls = _image_urls(messages or [])
        hashes = [hashlib.sha1(u.encode()).hexdigest() for u in urls]
        record["images"] = hashes
        with self._lock:
            new = [(h, u) for h, u in zip(hashes, urls) if h not in self._saved_images]
            if new:
                with gzip.open(self.images_path, "at", encoding="utf-8") as f:
                    for h, u in new:
                        f.write(json.dumps({"h": h, "url": u}) + "\n")
                        self._saved_images.add(h)
            self._index += 1
            with gzip.open(self.path, "at", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")))
                f.write("\n")
