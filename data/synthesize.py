"""Synthetic 'city photo' generator that emits YOLO-format training data.

Pastes reference images of cards / resources / workers / tokens / events onto a
varied background with realistic camera-style distortions:
  - 4-corner perspective warp (simulates off-axis camera)
  - drop shadow on cards/events (simulates table lighting)
  - background variety: real board photos + HSV-jittered recolours + solid colours
  - rotation, scale, lighting jitter, blur, partial occlusion

Outputs:
  datasets/entity_classifier/images/{train,val}/synth_{i}.jpg
  datasets/entity_classifier/labels/{train,val}/synth_{i}.txt   (YOLO: cls cx cy w h)

Class IDs match datasets/entity_classifier/data.yaml.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CLASS_IDS = {
    "card": 0,
    "resource": 1,
    "worker": 2,
    "token": 3,
    "event": 4,
}

SOURCE_DIRS = {
    "card": "card",
    "resource": "resource",
    "worker": "worker",
    "token": "token",
    "event": "event",
}

SKIP_PATTERNS = ("_group", "card_back", "_pair", "_compact")

# Categories that should get drop shadows (flat objects on a table)
SHADOW_CATEGORIES = {"card", "event"}


@dataclass
class PasteSpec:
    card: tuple[int, int] = (5, 14)
    resource: tuple[int, int] = (0, 8)
    worker: tuple[int, int] = (0, 4)
    token: tuple[int, int] = (0, 6)
    event: tuple[int, int] = (0, 3)

    def sample(self) -> dict[str, int]:
        return {k: random.randint(*getattr(self, k)) for k in CLASS_IDS}


# --------------------------------------------------------------------- helpers

def _list_reference_images(image_root: Path, class_name: str) -> list[Path]:
    src = image_root / SOURCE_DIRS[class_name]
    if not src.exists():
        return []
    out = []
    for p in src.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if any(skip in p.stem for skip in SKIP_PATTERNS):
            continue
        out.append(p)
    return out


def _jitter(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 1.30))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.30))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.80, 1.25))
    if random.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    return img


def _find_perspective_coeffs(
    target: list[tuple[float, float]],
    source: list[tuple[float, float]],
) -> list[float]:
    """Compute 8 PIL perspective coeffs that map output pixels back to source pixels.

    target = where each source corner appears in the output image.
    source = original source corner coords.
    PIL.Image.transform with PERSPECTIVE expects the inverse mapping (output->input).
    """
    A = []
    for s, t in zip(source, target):
        A.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        A.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    A = np.array(A, dtype=np.float64)
    B = np.array(source, dtype=np.float64).reshape(8)
    coeffs, *_ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs.tolist()


def _perspective_warp(
    img_rgba: Image.Image,
    max_offset_frac: float = 0.18,
) -> Image.Image:
    """Apply random 4-corner perspective distortion to an RGBA image.

    Each corner is independently nudged inward by up to max_offset_frac of the
    image dimensions, simulating a card photographed from an off-axis camera.
    Output canvas matches input dimensions; pixels outside the warped quad are
    transparent.
    """
    w, h = img_rgba.size
    max_dx = w * max_offset_frac
    max_dy = h * max_offset_frac

    src = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    dst = [
        (random.uniform(0, max_dx), random.uniform(0, max_dy)),
        (w - random.uniform(0, max_dx), random.uniform(0, max_dy)),
        (w - random.uniform(0, max_dx), h - random.uniform(0, max_dy)),
        (random.uniform(0, max_dx), h - random.uniform(0, max_dy)),
    ]
    coeffs = _find_perspective_coeffs(dst, src)
    return img_rgba.transform(
        (w, h),
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )


def _build_card(
    ref_path: Path,
    target_long: int,
    apply_perspective: bool,
) -> Image.Image | None:
    """Open a reference, scale, jitter, rotate, optionally perspective-warp.

    Returns an RGBA image with transparent areas outside the warped card.
    """
    try:
        img = Image.open(ref_path).convert("RGB")
    except Exception:
        return None

    long = max(img.size)
    scale = target_long / long
    new_w = max(8, int(img.size[0] * scale))
    new_h = max(8, int(img.size[1] * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    img = _jitter(img).convert("RGBA")

    angle = random.uniform(-22, 22)
    img = img.rotate(
        angle,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

    if apply_perspective:
        img = _perspective_warp(img, max_offset_frac=random.uniform(0.05, 0.22))

    return img


def _drop_shadow(
    rgba: Image.Image,
    blur: int = 12,
    offset: tuple[int, int] = (10, 14),
    opacity: float = 0.45,
) -> Image.Image:
    """Return a shadow image (RGBA) sized to fit the original alpha plus offset/blur padding."""
    alpha = rgba.split()[-1]
    pad = blur * 2 + max(abs(offset[0]), abs(offset[1])) + 4
    shadow = Image.new("L", (alpha.width + 2 * pad, alpha.height + 2 * pad), 0)
    shadow.paste(alpha, (pad + offset[0], pad + offset[1]))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))
    shadow = shadow.point(lambda v: int(v * opacity))
    out = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
    out.putalpha(shadow)
    return out


def _tight_bbox_from_alpha(rgba: Image.Image) -> tuple[int, int, int, int] | None:
    bbox = rgba.split()[-1].getbbox()
    return bbox  # (l, t, r, b) or None


# --------------------------------------------------------------- backgrounds

def _load_backgrounds(image_root: Path) -> list[Image.Image]:
    board_dir = image_root / "board"
    bgs = []
    if board_dir.exists():
        for p in board_dir.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    bgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
    return bgs


def _hsv_jitter(img: Image.Image) -> Image.Image:
    """Random HSV recolor — simulates different table colors / lighting tones."""
    arr = np.array(img.convert("HSV"), dtype=np.int16)
    arr[..., 0] = (arr[..., 0] + random.randint(-30, 30)) % 256
    arr[..., 1] = np.clip(arr[..., 1] * random.uniform(0.55, 1.25), 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * random.uniform(0.65, 1.20), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")


def _solid_or_gradient(size: tuple[int, int]) -> Image.Image:
    """Generate a solid-colour or vertical-gradient background."""
    W, H = size
    if random.random() < 0.5:
        # solid
        c = (random.randint(40, 200), random.randint(40, 200), random.randint(40, 200))
        return Image.new("RGB", size, c)
    # vertical gradient
    top = np.array([random.randint(40, 220), random.randint(40, 220), random.randint(40, 220)])
    bottom = np.array([random.randint(40, 220), random.randint(40, 220), random.randint(40, 220)])
    arr = np.linspace(top, bottom, H).astype(np.uint8)
    arr = np.repeat(arr[:, None, :], W, axis=1)
    return Image.fromarray(arr)


def _make_canvas(backgrounds: list[Image.Image], size: tuple[int, int]) -> Image.Image:
    r = random.random()
    if backgrounds and r < 0.55:
        bg = random.choice(backgrounds).copy().resize(size, Image.Resampling.BICUBIC)
        return bg
    if backgrounds and r < 0.85:
        bg = random.choice(backgrounds).copy().resize(size, Image.Resampling.BICUBIC)
        return _hsv_jitter(bg)
    return _solid_or_gradient(size)


def _final_canvas_jitter(canvas: Image.Image) -> Image.Image:
    """Small global jitter applied AFTER pasting -- vignette/blur/lighting shift."""
    if random.random() < 0.3:
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.2:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return canvas


# --------------------------------------------------------------- main pipeline

def _yolo_line(cls_id: int, x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> str:
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _pick_size(class_name: str, canvas_w: int) -> int:
    base = canvas_w / 1000.0
    return int(base * {
        "card": random.randint(170, 280),
        "resource": random.randint(45, 95),
        "worker": random.randint(55, 110),
        "token": random.randint(40, 85),
        "event": random.randint(110, 190),
    }[class_name])


def generate_one(
    canvas_size: tuple[int, int],
    references: dict[str, list[Path]],
    backgrounds: list[Image.Image],
    paste_spec: PasteSpec,
    max_iou: float = 0.45,
) -> tuple[Image.Image, list[str]]:
    W, H = canvas_size
    canvas = _make_canvas(backgrounds, canvas_size).convert("RGBA")
    placed_boxes: list[tuple[int, int, int, int]] = []
    labels: list[str] = []

    counts = paste_spec.sample()
    order = ["card", "event", "worker", "resource", "token"]

    for class_name in order:
        cls_id = CLASS_IDS[class_name]
        pool = references.get(class_name) or []
        if not pool:
            continue

        for _ in range(counts[class_name]):
            ref_path = random.choice(pool)
            target_long = _pick_size(class_name, W)
            apply_persp = (class_name in SHADOW_CATEGORIES) and random.random() < 0.85

            warped = _build_card(ref_path, target_long, apply_persp)
            if warped is None:
                continue

            tight = _tight_bbox_from_alpha(warped)
            if tight is None:
                continue
            tw = tight[2] - tight[0]
            th = tight[3] - tight[1]
            if tw < 8 or th < 8:
                continue

            placed = False
            for _attempt in range(15):
                # position so that the tight bbox lands inside the canvas
                px = random.randint(-tight[0], max(-tight[0], W - tight[2]))
                py = random.randint(-tight[1], max(-tight[1], H - tight[3]))
                bbox_world = (
                    tight[0] + px,
                    tight[1] + py,
                    tight[2] + px,
                    tight[3] + py,
                )
                if all(_iou(bbox_world, b) < max_iou for b in placed_boxes):
                    if class_name in SHADOW_CATEGORIES:
                        shadow = _drop_shadow(
                            warped,
                            blur=random.randint(6, 18),
                            offset=(random.randint(-6, 14), random.randint(2, 18)),
                            opacity=random.uniform(0.30, 0.55),
                        )
                        # paste shadow centered relative to the card's tight bbox
                        sx = px + (warped.width - shadow.width) // 2
                        sy = py + (warped.height - shadow.height) // 2
                        canvas.alpha_composite(shadow, (sx, sy))
                    canvas.alpha_composite(warped, (px, py))
                    placed_boxes.append(bbox_world)
                    labels.append(_yolo_line(cls_id, *bbox_world, W=W, H=H))
                    placed = True
                    break

            if not placed:
                # accept overlap rather than dropping
                px = random.randint(-tight[0], max(-tight[0], W - tight[2]))
                py = random.randint(-tight[1], max(-tight[1], H - tight[3]))
                bbox_world = (
                    tight[0] + px, tight[1] + py, tight[2] + px, tight[3] + py
                )
                canvas.alpha_composite(warped, (px, py))
                placed_boxes.append(bbox_world)
                labels.append(_yolo_line(cls_id, *bbox_world, W=W, H=H))

    canvas = canvas.convert("RGB")
    canvas = _final_canvas_jitter(canvas)
    return canvas, labels


def generate(
    n: int = 200,
    image_root: str | Path = "data/images",
    out_root: str | Path = "datasets/entity_classifier",
    canvas_size: tuple[int, int] = (1024, 1024),
    val_fraction: float = 0.1,
    paste_spec: PasteSpec | None = None,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    image_root = Path(image_root)
    out_root = Path(out_root)
    paste_spec = paste_spec or PasteSpec()

    references = {cls: _list_reference_images(image_root, cls) for cls in CLASS_IDS}
    missing = [c for c, paths in references.items() if not paths]
    if missing:
        print(f"WARN: no reference images found for: {missing}")

    backgrounds = _load_backgrounds(image_root)
    if not backgrounds:
        print("WARN: no board backgrounds found; using solid colors")

    n_val = max(1, int(n * val_fraction))
    splits = ["train"] * (n - n_val) + ["val"] * n_val
    random.shuffle(splits)

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, split in enumerate(splits):
        img, labels = generate_one(canvas_size, references, backgrounds, paste_spec)
        img_path = out_root / "images" / split / f"synth_{i:05d}.jpg"
        lbl_path = out_root / "labels" / split / f"synth_{i:05d}.txt"
        img.save(img_path, quality=92)
        lbl_path.write_text("\n".join(labels), encoding="utf-8")

    print(f"Generated {n} synthetic images at {out_root}")


if __name__ == "__main__":
    generate(n=300, seed=42)
