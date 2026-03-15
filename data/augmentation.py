from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

def random_lighting(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 1.35))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    return img

def random_blur(img: Image.Image) -> Image.Image:
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.8)))
    return img

def random_rotation_scale(img: Image.Image) -> Image.Image:
    angle = random.uniform(-20, 20)
    scale = random.uniform(0.75, 1.2)

    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    img = img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    return img

def random_perspective_like_crop(img: Image.Image, output_size=(256, 256)) -> Image.Image:
    # cheap approximation of camera framing changes
    w, h = img.size
    crop_ratio = random.uniform(0.7, 1.0)
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)

    left = random.randint(0, max(0, w - cw))
    top = random.randint(0, max(0, h - ch))
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize(output_size, Image.Resampling.BICUBIC)
    return img

def random_occlusion(img: Image.Image, max_boxes=3) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for _ in range(random.randint(1, max_boxes)):
        occ_w = random.randint(int(w * 0.08), int(w * 0.35))
        occ_h = random.randint(int(h * 0.08), int(h * 0.35))
        x1 = random.randint(0, w - occ_w)
        y1 = random.randint(0, h - occ_h)
        x2 = x1 + occ_w
        y2 = y1 + occ_h

        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([x1, y1, x2, y2], fill=color)

    return img

def partial_card_view(img: Image.Image, output_size=(256, 256)) -> Image.Image:
    # simulates only top / bottom / side visible
    w, h = img.size
    mode = random.choice(["top", "bottom", "left", "right", "center"])

    if mode == "top":
        crop = (0, 0, w, random.randint(int(h * 0.2), int(h * 0.6)))
    elif mode == "bottom":
        y1 = random.randint(int(h * 0.4), int(h * 0.8))
        crop = (0, y1, w, h)
    elif mode == "left":
        crop = (0, 0, random.randint(int(w * 0.2), int(w * 0.6)), h)
    elif mode == "right":
        x1 = random.randint(int(w * 0.4), int(w * 0.8))
        crop = (x1, 0, w, h)
    else:
        x1 = random.randint(0, int(w * 0.3))
        y1 = random.randint(0, int(h * 0.3))
        x2 = random.randint(int(w * 0.7), w)
        y2 = random.randint(int(h * 0.7), h)
        crop = (x1, y1, x2, y2)

    img = img.crop(crop)
    img = img.resize(output_size, Image.Resampling.BICUBIC)
    return img

def augment_card_image(image_path: str, output_dir: str = "data/training", n: int = 30, output_size=(256, 256)) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(image_path).convert("RGB")

    # save normalized original
    base = ImageOps.contain(original, output_size)
    canvas = Image.new("RGB", output_size, (0, 0, 0))
    x = (output_size[0] - base.width) // 2
    y = (output_size[1] - base.height) // 2
    canvas.paste(base, (x, y))
    canvas.save(output_dir / "original.jpg", quality=95)

    for i in range(n):
        img = original.copy()

        if random.random() < 0.8:
            img = random_rotation_scale(img)

        if random.random() < 0.8:
            img = random_lighting(img)

        if random.random() < 0.5:
            img = random_blur(img)

        # choose one framing style
        if random.random() < 0.5:
            img = partial_card_view(img, output_size=output_size)
        else:
            img = random_perspective_like_crop(img, output_size=output_size)

        if random.random() < 0.6:
            img = random_occlusion(img)

        img.save(output_dir / f"aug_{i:03d}.jpg", quality=95)