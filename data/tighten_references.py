"""Tighten reference images by cropping out background wood/desk.

Backs up originals to data/images_original/, then writes tight crops back
to data/images/. Idempotent: if a backup already exists for a file, that
file is treated as already-tightened and skipped.

Algorithm
  1. Downsample to ~800px wide (grabCut is ~quadratic in pixels).
  2. Run grabCut with a hint rectangle inset 5% from each edge.
  3. Morph-close the resulting foreground mask to fill small holes.
  4. Take the largest connected component as the card region.
  5. Scale its bounding box back to the original resolution.

grabCut iteratively builds GMM color models for foreground/background,
which handles textured wood much better than a fixed Lab threshold.

Usage
  python data/tighten_references.py --preview                # save annotated previews
  python data/tighten_references.py --dry-run                # report only, no writes
  python data/tighten_references.py                          # apply
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


SRC_ROOT = Path("data/images")
BACKUP_ROOT = Path("data/images_original")
PREVIEW_ROOT = Path("data/_tighten_preview")
PADDING = 4                # final inset around foreground bbox
GRABCUT_TARGET_W = 800     # downsample width for grabCut speed
GRABCUT_ITERS = 5
MORPH_KERNEL = 9
MIN_AREA_FRAC = 0.02

# Hint-rectangle inset per category. Small items (tokens, single resources,
# workers) need a tight central hint or grabCut latches onto the wood-grain
# texture/lighting instead of the item.
HINT_INSET_BY_DIR = {
    "card": 0.05,
    "event": 0.05,
    "worker": 0.32,
    "resource": 0.32,
    "token": 0.36,
}
DEFAULT_HINT_INSET = 0.10

# Skip board/journey reference photos -- those ARE real-photo backgrounds
SKIP_DIRS = {"board"}


def _grabcut_bbox(small: np.ndarray, hint_inset_frac: float) -> tuple[int, int, int, int] | None:
    Hs, Ws = small.shape[:2]
    mask = np.zeros((Hs, Ws), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    inset_x = max(2, int(Ws * hint_inset_frac))
    inset_y = max(2, int(Hs * hint_inset_frac))
    rect = (inset_x, inset_y, Ws - 2 * inset_x, Hs - 2 * inset_y)

    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[biggest, cv2.CC_STAT_AREA]
    if area < MIN_AREA_FRAC * Hs * Ws:
        return None

    xs = stats[biggest, cv2.CC_STAT_LEFT]
    ys = stats[biggest, cv2.CC_STAT_TOP]
    ws = stats[biggest, cv2.CC_STAT_WIDTH]
    hs = stats[biggest, cv2.CC_STAT_HEIGHT]
    return int(xs), int(ys), int(ws), int(hs)


def tight_bbox(img_bgr: np.ndarray, category: str = "") -> tuple[int, int, int, int] | None:
    H0, W0 = img_bgr.shape[:2]

    if W0 > GRABCUT_TARGET_W:
        scale = GRABCUT_TARGET_W / W0
        small = cv2.resize(img_bgr, (GRABCUT_TARGET_W, int(H0 * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = img_bgr.copy()

    inset = HINT_INSET_BY_DIR.get(category, DEFAULT_HINT_INSET)
    res = _grabcut_bbox(small, inset)

    # if grabCut fills nearly the entire hint rect, it likely failed -- the
    # hint rect was too generous and grabCut just kept the rect. Retry tighter.
    if res is not None:
        xs, ys, ws, hs = res
        Hs, Ws = small.shape[:2]
        hint_w = Ws - 2 * max(2, int(Ws * inset))
        hint_h = Hs - 2 * max(2, int(Hs * inset))
        if ws * hs > 0.93 * hint_w * hint_h and inset < 0.30:
            res = _grabcut_bbox(small, max(inset + 0.20, 0.30))

    if res is None:
        return None
    xs, ys, ws, hs = res

    x1 = max(0, int(xs / scale) - PADDING)
    y1 = max(0, int(ys / scale) - PADDING)
    x2 = min(W0, int((xs + ws) / scale) + PADDING)
    y2 = min(H0, int((ys + hs) / scale) + PADDING)
    return x1, y1, x2, y2


def _save_preview(src_path: Path, img: np.ndarray, bbox: tuple[int, int, int, int]) -> Path:
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    out = img.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), max(2, img.shape[1] // 200))
    p = PREVIEW_ROOT / f"{src_path.parent.name}__{src_path.name}"
    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return p


def process_one(src_path: Path, dry_run: bool, preview: bool) -> tuple[str, tuple]:
    if src_path.parent.name in SKIP_DIRS:
        return "skipped (board/)", ()

    rel = src_path.relative_to(SRC_ROOT)
    backup_path = BACKUP_ROOT / rel
    if backup_path.exists():
        return "skipped (backup exists)", ()

    img = cv2.imread(str(src_path))
    if img is None:
        return "failed (could not read)", ()

    bbox = tight_bbox(img, category=src_path.parent.name)
    if bbox is None:
        return "skipped (no clear foreground)", ()

    x1, y1, x2, y2 = bbox
    H0, W0 = img.shape[:2]
    cw, ch = x2 - x1, y2 - y1
    pct_kept = 100 * (cw * ch) / (H0 * W0)

    if preview:
        p = _save_preview(src_path, img, bbox)
        return f"preview {W0}x{H0} -> {cw}x{ch} ({pct_kept:.0f}%) at {p.name}", bbox

    if dry_run:
        return f"would crop {W0}x{H0} -> {cw}x{ch} ({pct_kept:.0f}%)", bbox

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, backup_path)
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(str(src_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return f"cropped {W0}x{H0} -> {cw}x{ch} ({pct_kept:.0f}%)", bbox


def main(dry_run: bool, preview: bool, limit: int | None) -> None:
    if not SRC_ROOT.exists():
        raise SystemExit(f"missing source: {SRC_ROOT}")

    paths = sorted(p for p in SRC_ROOT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        paths = paths[:limit]
    print(f"processing {len(paths)} reference images")

    counts: dict[str, int] = {}
    for p in paths:
        status, _ = process_one(p, dry_run=dry_run, preview=preview)
        kind = status.split(" ", 1)[0]
        counts[kind] = counts.get(kind, 0) + 1
        if counts[kind] <= 4:
            print(f"  {p.relative_to(SRC_ROOT)}: {status}")

    print("---")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if dry_run:
        print("(dry run -- nothing written)")
    if preview:
        print(f"(preview images at {PREVIEW_ROOT})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="write green-rectangle previews to data/_tighten_preview/")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(dry_run=args.dry_run, preview=args.preview, limit=args.limit)
