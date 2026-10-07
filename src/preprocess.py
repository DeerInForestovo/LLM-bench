"""Data preprocessing pipeline for LLM-bench.

Raw logo assets (data/raw/) are heterogeneous: different resolutions,
color modes (palette / RGB / RGBA / JPEG-derived), and background matting
(white, black, or vendor-tinted). This pipeline normalizes every sample
into a clean, background-transparent 256x256 RGBA PNG so that downstream
measurements are not confounded by acquisition artifacts.

Pipeline stages per sample:
  1. Mode normalization  -> convert to RGBA (depalettize if necessary).
  2. Background removal  -> border-seeded flood fill with color tolerance.
     A border color is only treated as a *matte background* when it is
     near-white or near-black; chromatic border regions (e.g. app-icon
     backplates that are part of the visual identity) are preserved.
  3. Halo suppression    -> strip the 1-px anti-aliased matte fringe left
     around the foreground after flood filling.
  4. Geometry normalization -> alpha-bbox trim, aspect-preserving resize
     onto a transparent 256x256 canvas (LANCZOS).

A JSON report with per-sample statistics is written to
results/preprocess_report.json.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORT_PATH = Path(__file__).resolve().parent.parent / "results" / "preprocess_report.json"

CANVAS_SIZE = 256
FLOOD_TOLERANCE = 28          # per-channel L-inf distance for matte matching
MATTE_LIGHT_THRESH = 232      # >= this on all channels -> near-white matte
MATTE_DARK_THRESH = 24        # <= this on all channels -> near-black matte


def dominant_border_color(arr: np.ndarray) -> np.ndarray:
    """Return the median color of border pixels (opaque ones only)."""
    h, w = arr.shape[:2]
    border = np.concatenate(
        [arr[0, :, :3], arr[h - 1, :, :3], arr[:, 0, :3], arr[:, w - 1, :3]]
    )
    alpha = np.concatenate(
        [arr[0, :, 3], arr[h - 1, :, 3], arr[:, 0, 3], arr[:, w - 1, 3]]
    )
    opaque = border[alpha > 0]
    if len(opaque) == 0:
        return np.array([255, 255, 255])
    return np.median(opaque, axis=0)


def is_matte(color: np.ndarray) -> bool:
    """A border color only counts as a removable matte if it is neutral:
    near-white or near-black. Chromatic backplates are part of the logo."""
    return bool(
        np.all(color >= MATTE_LIGHT_THRESH) or np.all(color <= MATTE_DARK_THRESH)
    )


def flood_fill_background(arr: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """Remove the matte background via border-seeded BFS flood fill.

    Pixels connected to the border whose color is within FLOOD_TOLERANCE
    (L-inf) of the matte color become fully transparent.
    """
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3].astype(np.int16)
    target = matte.astype(np.int16)
    match = np.all(np.abs(rgb - target) <= FLOOD_TOLERANCE, axis=2)

    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    for x in range(w):
        for y in (0, h - 1):
            if match[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if match[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and match[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    arr[visited, 3] = 0
    return arr


def suppress_halo(arr: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """Clear the anti-aliased matte fringe: opaque pixels that touch a
    transparent pixel and are still close to the matte color (with a
    relaxed tolerance) are made transparent as well."""
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3].astype(np.int16)
    target = matte.astype(np.int16)

    transparent = alpha == 0
    near_transparent = np.zeros_like(transparent)
    near_transparent[1:, :] |= transparent[:-1, :]
    near_transparent[:-1, :] |= transparent[1:, :]
    near_transparent[:, 1:] |= transparent[:, :-1]
    near_transparent[:, :-1] |= transparent[:, 1:]

    close = np.all(np.abs(rgb - target) <= FLOOD_TOLERANCE * 2.5, axis=2)
    fringe = near_transparent & close & (alpha > 0)
    arr[fringe, 3] = 0
    return arr


def normalize_geometry(img: Image.Image) -> Image.Image:
    """Trim transparent margins and paste, aspect-preserved and centered,
    onto a transparent CANVAS_SIZE x CANVAS_SIZE canvas with a small margin."""
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)

    margin = 8
    target = CANVAS_SIZE - 2 * margin
    scale = target / max(img.size)
    new_size = (
        max(1, round(img.width * scale)),
        max(1, round(img.height * scale)),
    )
    img = img.resize(new_size, Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    offset = ((CANVAS_SIZE - img.width) // 2, (CANVAS_SIZE - img.height) // 2)
    canvas.paste(img, offset, img)
    return canvas


def preprocess_file(raw_path: Path, out_path: Path) -> dict:
    img = Image.open(raw_path).convert("RGBA")
    arr = np.array(img)
    original_size = img.size

    matte = dominant_border_color(arr)
    removed = False
    if is_matte(matte):
        arr = flood_fill_background(arr, matte)
        arr = suppress_halo(arr, matte)
        removed = True

    img = Image.fromarray(arr, "RGBA")
    img = normalize_geometry(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {
        "file": raw_path.name,
        "original_size": list(original_size),
        "matte_color": [int(c) for c in matte],
        "background_removed": removed,
        "output_size": list(img.size),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for raw_path in sorted(RAW_DIR.glob("*.png")):
        out_path = OUT_DIR / raw_path.name
        stats = preprocess_file(raw_path, out_path)
        report.append(stats)
        print(f"[preprocess] {raw_path.name}: matte={stats['matte_color']} "
              f"removed={stats['background_removed']} -> {out_path.name}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[preprocess] report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
