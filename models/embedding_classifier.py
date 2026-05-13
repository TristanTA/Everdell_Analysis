"""Stage-2 identifier: pretrained image embedding + nearest-neighbor lookup.

A new card or token is added by dropping a reference image into the indexed
folder and re-running build_index() — no retraining. One reference per class
is enough; multiple references improve robustness.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import models, transforms


def _rotate(img: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return img
    return img.rotate(
        angle,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def _jitter(img: Image.Image) -> Image.Image:
    """Small in-place lighting/rotation jitter applied on top of a base orientation."""
    a = img.copy()
    if random.random() < 0.7:
        a = _rotate(a, int(random.uniform(-8, 8)))
    if random.random() < 0.85:
        a = ImageEnhance.Brightness(a).enhance(random.uniform(0.75, 1.25))
        a = ImageEnhance.Contrast(a).enhance(random.uniform(0.85, 1.20))
        a = ImageEnhance.Color(a).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.30:
        a = a.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return a


def _augment_for_index(img: Image.Image, n: int) -> list[Image.Image]:
    """Generate n in-memory variants of a reference image.

    For n >= 4, all four cardinal orientations (0/90/180/270) are included as
    base anchors so the index is robust to camera-rotation in the original
    references (~70% of our card photos are landscape because the phone was
    held sideways). Each base orientation also contributes (n//4) lightly
    jittered copies for lighting/small-angle robustness.
    """
    if n <= 1:
        return [img]

    if n >= 4:
        bases = [_rotate(img, a) for a in (0, 90, 180, 270)]
        per_base = n // 4
        extra = n % 4
    else:
        bases = [img]
        per_base = n
        extra = 0

    out: list[Image.Image] = []
    for i, base in enumerate(bases):
        count = per_base + (1 if i < extra else 0)
        out.append(base)  # the cardinal-rotation anchor itself
        for _ in range(count - 1):
            out.append(_jitter(base))
    return out


def _default_label_fn(path: Path) -> str:
    """Strip Everdell variant suffixes so all variants of a card map to one label.

    architect.jpg                    -> architect
    architect_resources_3.jpg        -> architect
    bard_tokens_2.jpg                -> bard
    coin_3+1_metal.jpg               -> coin_3+1_metal   (no known suffix -> stem)
    a_brilliant_marketing_plan_resources_2.jpg -> a_brilliant_marketing_plan
    """
    stem = path.stem
    # known variant suffixes seen in data/images/
    stem = re.sub(
        r"_(tokens?|resources?|cards?|workers?|coins?|points?)_\d+(_\w+)?$",
        "",
        stem,
    )
    stem = re.sub(r"\d+$", "", stem)  # trailing duplicates: carnival2, gatherer3
    return stem.rstrip("_")


@dataclass
class Match:
    label: str
    similarity: float
    source: str  # filename of the reference that matched


class EmbeddingClassifier:
    """Wrap a pretrained ImageNet backbone, expose embed() and predict()."""

    def __init__(
        self,
        backbone: str = "resnet50",
        device: str | None = None,
        weights: str = "DEFAULT",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_name = backbone
        self.model = self._load_backbone(backbone, weights).to(self.device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.embeddings: torch.Tensor | None = None  # (N, D) L2-normalized
        self.labels: list[str] = []
        self.sources: list[str] = []

    @staticmethod
    def _load_backbone(name: str, weights: str) -> torch.nn.Module:
        if name == "resnet50":
            m = models.resnet50(weights=weights)
            m.fc = torch.nn.Identity()
            return m
        if name == "resnet18":
            m = models.resnet18(weights=weights)
            m.fc = torch.nn.Identity()
            return m
        raise ValueError(f"unknown backbone: {name}")

    @torch.inference_mode()
    def embed(self, image: Image.Image | str | Path) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model(x)
        feat = F.normalize(feat, dim=1)
        return feat.squeeze(0).cpu()

    def build_index(
        self,
        image_dir: str | Path,
        label_fn: Callable[[Path], str] | None = None,
        recursive: bool = False,
        exts: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        skip_substrings: tuple[str, ...] = ("_group", "card_back"),
        augment_per_image: int = 1,
        augment_seed: int | None = 0,
    ) -> int:
        """Build the embedding index.

        augment_per_image: if > 1, generate this many in-memory augmented
        copies of each reference (mild rotation, lighting, blur) and embed
        them all. Each augmented copy keeps the same label, so the index
        gains multiple anchors per card without any extra photo capture.
        """
        image_dir = Path(image_dir)
        label_fn = label_fn or _default_label_fn
        paths = (image_dir.rglob("*") if recursive else image_dir.iterdir())
        files = [
            p for p in paths
            if p.suffix.lower() in exts
            and not any(s in p.stem for s in skip_substrings)
        ]
        if not files:
            raise FileNotFoundError(f"No reference images in {image_dir}")

        if augment_per_image > 1 and augment_seed is not None:
            random.seed(augment_seed)

        embs, labels, sources = [], [], []
        for p in files:
            try:
                base_img = Image.open(p).convert("RGB")
                variants = (
                    _augment_for_index(base_img, augment_per_image)
                    if augment_per_image > 1 else [base_img]
                )
                lbl = label_fn(p)
                for i, v in enumerate(variants):
                    embs.append(self.embed(v))
                    labels.append(lbl)
                    sources.append(p.name if i == 0 else f"{p.name}#aug{i}")
            except Exception as exc:
                print(f"WARN: failed to embed {p.name}: {exc}")

        self.embeddings = torch.stack(embs)
        self.labels = labels
        self.sources = sources
        return len(embs)

    def predict(self, image: Image.Image | str | Path, top_k: int = 3) -> list[Match]:
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")
        q = self.embed(image).unsqueeze(0)  # (1, D)
        sims = (self.embeddings @ q.T).squeeze(1)  # cosine, since both are L2-normed
        k = min(top_k, sims.numel())
        vals, idxs = torch.topk(sims, k)
        return [
            Match(label=self.labels[int(i)], similarity=float(v), source=self.sources[int(i)])
            for v, i in zip(vals.tolist(), idxs.tolist())
        ]

    def predict_aggregated(
        self,
        image: Image.Image | str | Path,
        top_k: int = 5,
    ) -> list[Match]:
        """Top-k single matches collapsed to best-similarity-per-label.

        Pulls a generous raw-candidate pool so that augmented variants of the
        wrong card can't crowd the correct card's bare reference out of the
        collapse step.
        """
        # Fetch enough raw candidates that every distinct label has a fair
        # chance to be represented even with heavy index-time augmentation.
        n_unique = len(set(self.labels)) if self.labels else 0
        raw_k = max(top_k * 8, 64, min(n_unique, 200))
        if self.embeddings is not None:
            raw_k = min(raw_k, self.embeddings.shape[0])
        raw = self.predict(image, top_k=raw_k)
        best: dict[str, Match] = {}
        for m in raw:
            cur = best.get(m.label)
            if cur is None or m.similarity > cur.similarity:
                best[m.label] = m
        return sorted(best.values(), key=lambda m: m.similarity, reverse=True)[:top_k]

    def save(self, path: str | Path) -> None:
        if self.embeddings is None:
            raise RuntimeError("Nothing to save: index not built.")
        torch.save(
            {
                "embeddings": self.embeddings,
                "labels": self.labels,
                "sources": self.sources,
                "backbone": self.backbone_name,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.embeddings = blob["embeddings"]
        self.labels = blob["labels"]
        self.sources = blob["sources"]
