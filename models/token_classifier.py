"""Token (coin) classifier.

Coins come in denominations 1 and 3, in metal and cardboard variants.
Filenames like coin_1_metal.jpg, coin_3_cardboard.jpg encode the class
directly, so we keep the full stem as the label.
"""
from __future__ import annotations

import re
from pathlib import Path

from .embedding_classifier import EmbeddingClassifier, Match


def _token_label(path: Path) -> str:
    # coin_3+1_metal -> coin_3 (treat compound stacks as their face)
    stem = path.stem
    m = re.match(r"(coin)_(\d+)", stem)
    if m:
        return f"coin_{m.group(2)}"
    return stem


class TokenClassifier:
    DEFAULT_DIR = Path("data/images/token")
    DEFAULT_INDEX = Path("models/index/tokens.pt")
    # numeric value of each label, used by scoring
    DENOMINATIONS = {"coin_1": 1, "coin_3": 3}

    def __init__(self, backbone: str = "resnet18", device: str | None = None):
        # resnet18 is plenty for the small token visual variety
        self.clf = EmbeddingClassifier(backbone=backbone, device=device)

    def build(
        self,
        image_dir: str | Path | None = None,
        save_to: str | Path | None = None,
        augment_per_image: int = 8,
    ) -> int:
        n = self.clf.build_index(
            image_dir or self.DEFAULT_DIR,
            label_fn=_token_label,
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

    def value_of(self, label: str) -> int:
        return self.DENOMINATIONS.get(label, 0)
