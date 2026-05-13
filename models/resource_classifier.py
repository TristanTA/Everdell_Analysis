"""Resource classifier: identifies twig / resin / berry / pebble.

Each visible resource crop is mapped to one of four types via the same
embedding-NN approach. For 'pile' crops we don't try to count individual
resources here; counting is left to the scoring stage which does not need
exact resource counts to compute city points.
"""
from __future__ import annotations

import re
from pathlib import Path

from .embedding_classifier import EmbeddingClassifier, Match


RESOURCE_TYPES = ("twig", "resin", "berry", "pebble")


def _resource_label(path: Path) -> str:
    # berry_up.jpg / berry_down.jpg / berry_group.jpg -> berry
    stem = path.stem.lower()
    for r in RESOURCE_TYPES:
        if stem.startswith(r):
            return r
    # fallback: first token
    return re.split(r"[_\d]", stem, maxsplit=1)[0]


class ResourceClassifier:
    DEFAULT_DIR = Path("data/images/resource")
    DEFAULT_INDEX = Path("models/index/resources.pt")

    def __init__(self, backbone: str = "resnet18", device: str | None = None):
        self.clf = EmbeddingClassifier(backbone=backbone, device=device)

    def build(
        self,
        image_dir: str | Path | None = None,
        save_to: str | Path | None = None,
        augment_per_image: int = 8,
    ) -> int:
        n = self.clf.build_index(
            image_dir or self.DEFAULT_DIR,
            label_fn=_resource_label,
            skip_substrings=(),  # keep _group images here; resource piles are a real class
            augment_per_image=augment_per_image,
        )
        if save_to:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            self.clf.save(save_to)
        return n

    def load(self, path: str | Path | None = None) -> None:
        self.clf.load(path or self.DEFAULT_INDEX)

    def identify(self, crop, top_k: int = 2) -> list[Match]:
        return self.clf.predict_aggregated(crop, top_k=top_k)
