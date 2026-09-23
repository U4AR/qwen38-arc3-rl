"""Optional `watch_video` tool: replay every rendered frame of recent actions.

The Duck normally only shows the model the final frame after each action.
Many ARC-AGI-3 actions produce a short multi-frame animation (things sliding,
blinking, a level-transition flash) whose intermediate frames are dropped.
This tool renders those frames as one labelled filmstrip image so a
vision-language model can look at them when it is genuinely stuck.

Enable with ``VIDEO_TOOL=1``.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image, ImageDraw

from inference.agent.vision_context import ARC_COLOR_MAP

VIDEO_TOOL_NAME = "watch_video"
_MAX_ACTIONS = 8
_MAX_FRAMES = 16
_CELL_SCALE = 4  # 64x64 grid -> 256x256 px tile
_LABEL_HEIGHT = 14
_COLUMNS = 4


def video_tool_enabled() -> bool:
    return os.environ.get("VIDEO_TOOL", "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RecordedAction:
    """All frames rendered by one environment action (animation + final)."""

    action: str
    step: int
    level: int
    frames: tuple[tuple[tuple[int, ...], ...], ...]


VIDEO_TOOL_DESCRIPTION = (
    "OPTIONAL and costly -- only call this when you are totally unclear what is happening or what to do "
    "(e.g. an action changed the board in a way your Python diffs cannot explain, or you have no working "
    "hypothesis for the goal). Returns one filmstrip IMAGE showing every rendered frame, including the "
    "intermediate animation frames that `history` omits, for the most recent actions. Each tile is labelled "
    "with the action, step and frame index. Prefer the `python` tool for everything else."
)

VIDEO_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VIDEO_TOOL_NAME,
        "description": VIDEO_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "last_actions": {
                    "type": "integer",
                    "description": f"How many of the most recent actions to replay (1-{_MAX_ACTIONS}, default 2).",
                },
            },
            "required": [],
        },
    },
}

VIDEO_TOOL_PROMPT_ADDENDUM = (
    "\n\nOptional video replay:\n"
    f"- A second tool, `{VIDEO_TOOL_NAME}`, shows an image of every rendered frame (including intermediate animation frames) "
    "for the last few actions.\n"
    "- It is expensive. Only use it when you are totally unclear what is going on or what to do next; "
    "otherwise rely on `python`.\n"
)


def _grid_tile(grid: Sequence[Sequence[int]], label: str) -> Image.Image:
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    board = Image.new("RGB", (max(1, cols), max(1, rows)), ARC_COLOR_MAP[0])
    px = board.load()
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            px[c, r] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
    board = board.resize((board.width * _CELL_SCALE, board.height * _CELL_SCALE), Image.Resampling.NEAREST)
    tile = Image.new("RGB", (board.width, board.height + _LABEL_HEIGHT), (255, 255, 255))
    tile.paste(board, (0, _LABEL_HEIGHT))
    ImageDraw.Draw(tile).text((2, 1), label, fill=(0, 0, 0))
    return tile


def select_frames(
    recorded: Sequence[RecordedAction], last_actions: int
) -> list[tuple[str, tuple[tuple[int, ...], ...]]]:
    """Pick (label, grid) pairs for the most recent actions, newest frames kept on overflow."""
    n = max(1, min(_MAX_ACTIONS, int(last_actions or 2)))
    picked: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    for entry in recorded[-n:]:
        total = len(entry.frames)
        for i, grid in enumerate(entry.frames):
            picked.append((f"step {entry.step} {entry.action} f{i + 1}/{total}", grid))
    return picked[-_MAX_FRAMES:]


def render_filmstrip(frames: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> Image.Image:
    tiles = [_grid_tile(grid, label) for label, grid in frames]
    width = max(t.width for t in tiles)
    height = max(t.height for t in tiles)
    cols = min(_COLUMNS, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    gap = 4
    sheet = Image.new("RGB", (cols * width + (cols - 1) * gap, rows * height + (rows - 1) * gap), (90, 90, 90))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * (width + gap), (i // cols) * (height + gap)))
    return sheet


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_video_response(
    recorded: Sequence[RecordedAction], arguments: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Return (tool-result text, image content part or None)."""
    try:
        last_actions = int(arguments.get("last_actions", 2) or 2)
    except (TypeError, ValueError):
        last_actions = 2
    frames = select_frames(recorded, last_actions)
    if not frames:
        return "No actions have been taken yet, so there is no video to show.", None
    labels = [label for label, _ in frames]
    text = (
        f"Filmstrip of {len(frames)} frame(s) attached in the next message, left-to-right, top-to-bottom: "
        + "; ".join(labels)
    )
    part = {"type": "image_url", "image_url": {"url": image_data_url(render_filmstrip(frames))}}
    return text, part
