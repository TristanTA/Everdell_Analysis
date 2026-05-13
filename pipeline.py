"""End-to-end Everdell city scorer.

  detect entities (YOLO)  ->  identify each crop (embedding NN)
                                |
                                v
                     classify token-on-card associations
                                |
                                v
                      build City  ->  EverdellScorer

Usage:
    from pipeline import EverdellPipeline
    pipe = EverdellPipeline.load_default()
    result = pipe.score_image("path/to/city.jpg")
    print(result["total"], result["breakdown"])

Before first use you must build the embedding indexes:
    EverdellPipeline.build_indexes()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image

from models import (
    CardIdentifier,
    EventIdentifier,
    ObjectDetector,
    ResourceClassifier,
    TokenClassifier,
)
from scoring import City, CityCard, EverdellScorer, ScoreBreakdown


INDEX_DIR = Path("models/index")
DEFAULT_DETECTOR_WEIGHTS = "yolo11s.pt"          # swap with your trained best.pt
DEFAULT_DETECTION_CONF = 0.30
DEFAULT_MIN_CARD_SIM = 0.55                      # below this -> 'unknown'


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _contains(card_bbox: list[float], token_bbox: list[float]) -> bool:
    """Token's center lies inside the card's bbox."""
    cx, cy = _bbox_center(token_bbox)
    return card_bbox[0] <= cx <= card_bbox[2] and card_bbox[1] <= cy <= card_bbox[3]


@dataclass
class IdentifiedItem:
    class_name: str           # 'card' / 'token' / etc., from the detector
    bbox: list[float]
    label: str | None         # specific identity, e.g. 'farm', 'coin_3'
    similarity: float = 0.0


@dataclass
class PipelineResult:
    items: list[IdentifiedItem] = field(default_factory=list)
    city: City = field(default_factory=City)
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    @property
    def total(self) -> int:
        return self.breakdown.total

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "breakdown": self.breakdown.__dict__,
            "items": [it.__dict__ for it in self.items],
        }


class EverdellPipeline:
    def __init__(
        self,
        detector: ObjectDetector,
        cards: CardIdentifier,
        tokens: TokenClassifier,
        resources: ResourceClassifier,
        events: EventIdentifier,
        scorer: EverdellScorer,
    ):
        self.detector = detector
        self.cards = cards
        self.tokens = tokens
        self.resources = resources
        self.events = events
        self.scorer = scorer

    # ---------------------------------------------------------------- factory

    @classmethod
    def load_default(
        cls,
        detector_weights: str = DEFAULT_DETECTOR_WEIGHTS,
        index_dir: Path = INDEX_DIR,
        db_path: str = "everdell_cards.db",
    ) -> "EverdellPipeline":
        detector = ObjectDetector(detector_weights)
        cards = CardIdentifier()
        cards.load(index_dir / "cards.pt")
        tokens = TokenClassifier()
        tokens.load(index_dir / "tokens.pt")
        resources = ResourceClassifier()
        resources.load(index_dir / "resources.pt")
        events = EventIdentifier()
        events.load(index_dir / "events.pt")
        scorer = EverdellScorer(db_path)
        return cls(detector, cards, tokens, resources, events, scorer)

    @staticmethod
    def build_indexes(
        image_root: str | Path = "data/images",
        index_dir: str | Path = INDEX_DIR,
    ) -> None:
        """Build and save all four embedding indexes from reference images."""
        image_root = Path(image_root)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        for cls, sub, out in [
            (CardIdentifier(), "card", "cards.pt"),
            (TokenClassifier(), "token", "tokens.pt"),
            (ResourceClassifier(), "resource", "resources.pt"),
            (EventIdentifier(), "event", "events.pt"),
        ]:
            n = cls.build(image_root / sub, save_to=index_dir / out)
            print(f"  built {out}: {n} reference images")

    # ----------------------------------------------------------------- scoring

    def identify_all(
        self,
        image: str | Path | Image.Image,
        det_conf: float = DEFAULT_DETECTION_CONF,
        min_card_sim: float = DEFAULT_MIN_CARD_SIM,
    ) -> list[IdentifiedItem]:
        detections = self.detector.detect(image, conf=det_conf)
        items: list[IdentifiedItem] = []
        for det in detections:
            cls_name = det["class_name"]
            crop = det["crop"]
            label, sim = None, 0.0

            if cls_name == "card":
                matches = self.cards.identify(crop, top_k=1)
                if matches and matches[0].similarity >= min_card_sim:
                    label, sim = matches[0].label, matches[0].similarity
            elif cls_name == "token":
                matches = self.tokens.identify(crop, top_k=1)
                if matches:
                    label, sim = matches[0].label, matches[0].similarity
            elif cls_name == "resource":
                matches = self.resources.identify(crop, top_k=1)
                if matches:
                    label, sim = matches[0].label, matches[0].similarity
            elif cls_name == "event":
                matches = self.events.identify(crop, top_k=1)
                if matches and matches[0].similarity >= min_card_sim:
                    label, sim = matches[0].label, matches[0].similarity
            elif cls_name == "worker":
                label = "worker"

            items.append(IdentifiedItem(
                class_name=cls_name,
                bbox=det["bbox"],
                label=label,
                similarity=sim,
            ))
        return items

    def assemble_city(self, items: list[IdentifiedItem]) -> City:
        cards: list[CityCard] = []
        events: list[str] = []
        workers_on_board = 0

        # spatial association: tokens whose center sits inside a card bbox
        # contribute their face value to that card's point_tokens.
        card_items = [it for it in items if it.class_name == "card" and it.label]
        token_items = [it for it in items if it.class_name == "token" and it.label]

        # init card rows
        rows = {id(it): CityCard(name=it.label) for it in card_items}

        for tok in token_items:
            value = self.tokens.value_of(tok.label)
            if value == 0:
                continue
            host = next(
                (c for c in card_items if _contains(c.bbox, tok.bbox)),
                None,
            )
            if host is not None:
                rows[id(host)].point_tokens += value

        cards = list(rows.values())

        for it in items:
            if it.class_name == "event" and it.label:
                events.append(it.label)
            elif it.class_name == "worker":
                workers_on_board += 1

        return City(
            cards=cards,
            events=events,
            journey_workers=0,  # set this from board-region analysis when ready
        )

    def score_image(
        self,
        image: str | Path | Image.Image,
        det_conf: float = DEFAULT_DETECTION_CONF,
    ) -> PipelineResult:
        items = self.identify_all(image, det_conf=det_conf)
        city = self.assemble_city(items)
        breakdown = self.scorer.score(city)
        return PipelineResult(items=items, city=city, breakdown=breakdown)


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", help="path to a city photo")
    p.add_argument("--build", action="store_true", help="rebuild reference indexes")
    p.add_argument("--weights", default=DEFAULT_DETECTOR_WEIGHTS)
    p.add_argument("--db", default="everdell_cards.db")
    args = p.parse_args()

    if args.build:
        EverdellPipeline.build_indexes()

    if args.image:
        pipe = EverdellPipeline.load_default(
            detector_weights=args.weights, db_path=args.db
        )
        result = pipe.score_image(args.image)
        print(json.dumps(result.to_dict(), indent=2, default=str))
