"""Card identifier: maps a cropped card image to a card name.

Wraps EmbeddingClassifier with sensible defaults for data/images/card/.
The default label function strips variant suffixes so e.g. `bard.jpg` and
`bard_tokens_2.jpg` both contribute to the 'bard' class.
"""
from __future__ import annotations

from pathlib import Path

from .embedding_classifier import EmbeddingClassifier, Match


class CardIdentifier:
    DEFAULT_DIR = Path("data/images/card")
    DEFAULT_INDEX = Path("models/index/cards.pt")

    def __init__(self, backbone: str = "resnet50", device: str | None = None):
        self.clf = EmbeddingClassifier(backbone=backbone, device=device)

    def build(
        self,
        image_dir: str | Path | None = None,
        save_to: str | Path | None = None,
        augment_per_image: int = 12,
    ) -> int:
        n = self.clf.build_index(
            image_dir or self.DEFAULT_DIR,
            augment_per_image=augment_per_image,
        )
        if save_to:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            self.clf.save(save_to)
        return n

    def load(self, path: str | Path | None = None) -> None:
        self.clf.load(path or self.DEFAULT_INDEX)

    def identify(self, crop, top_k: int = 3) -> list[Match]:
        return self.clf.predict_aggregated(crop, top_k=top_k)
