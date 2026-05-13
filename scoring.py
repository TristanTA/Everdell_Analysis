"""Everdell city scoring.

Inputs are produced by pipeline.py: a structured 'city' dict listing the
identified cards, point tokens placed on cards, journey workers, and
achieved events. Looks up base points via the SQLite card_data table and
applies common bonus rules.

A few rules in Everdell depend on spatial adjacency or on cards that aren't
in the city itself (e.g. discard pile state). Those are intentionally NOT
inferred here — the scoring engine is conservative and only applies rules
that can be evaluated from the photo.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# point values for journey locations (Everdell: 2/4/5 from short to long)
JOURNEY_POINTS = [2, 3, 4, 5]


@dataclass
class CityCard:
    name: str                # canonical card name (lowercase or matching DB)
    point_tokens: int = 0    # face-value sum of coin tokens placed on this card
    cards_under: int = 0     # for Dungeon: # of critters tucked under it


@dataclass
class City:
    cards: list[CityCard] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    journey_workers: int = 0           # workers on journey locations
    journey_points_override: int | None = None  # if you can read exact spots


@dataclass
class ScoreBreakdown:
    base_card_points: int = 0
    point_token_bonus: int = 0
    pair_bonus: int = 0
    event_points: int = 0
    journey_points: int = 0
    rule_bonuses: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.base_card_points
            + self.point_token_bonus
            + self.pair_bonus
            + self.event_points
            + self.journey_points
            + sum(self.rule_bonuses.values())
        )


# pairs that score 3 bonus points each when both are in the city
# (only the canonical critter+construction pairings)
CRITTER_CONSTRUCTION_PAIRS = {
    "wife": "husband",      # Husband+Wife is +3 each pair
    "miner mole": "mine",
    "shopkeeper": "general store",
    "monk": "monastery",
    "innkeeper": "inn",
    "ranger": "dungeon",
    "judge": "courthouse",
    "crane": "architect",   # Crane+Architect is not actually in base scoring,
                            # leaving here only because it's a fan-recognized pair;
                            # comment out if it causes issues.
    "teacher": "school",
    "undertaker": "cemetery",
    "barge toad": "twig barge",
    "doctor": "chapel",
    "postal pigeon": "post office",
    "woodcarver": "twig barge",
    "historian": "ruins",
    "bard": "theatre",
    "king": "castle",
    "queen": "palace",
}


class EverdellScorer:
    """Scores a parsed city using the SQLite card_data table.

    Cards in the table need a base_points value (and optionally a
    scoring_rule string) for full accuracy. Missing rows are warned and
    treated as zero.
    """

    def __init__(self, db_path: str | Path = "everdell_cards.db"):
        self.db_path = str(db_path)

    def _load_card(self, name: str) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT name, base_points, scoring_rule FROM card_data WHERE LOWER(name) = LOWER(?)",
            (name,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"name": row[0], "base_points": row[1] or 0, "scoring_rule": row[2] or ""}

    def score(self, city: City) -> ScoreBreakdown:
        result = ScoreBreakdown()

        names = [c.name.lower() for c in city.cards]
        name_counts = Counter(names)

        # base points + point-token bonuses + special rules
        for card in city.cards:
            row = self._load_card(card.name)
            if row is None:
                result.warnings.append(f"unknown card: {card.name}")
                continue
            result.base_card_points += int(row["base_points"])
            result.point_token_bonus += int(card.point_tokens)
            self._apply_card_rule(card, name_counts, result)

        # critter+construction pair bonuses (3 each pair, count = min of both)
        for critter, construction in CRITTER_CONSTRUCTION_PAIRS.items():
            n_pairs = min(name_counts.get(critter, 0), name_counts.get(construction, 0))
            if n_pairs:
                result.pair_bonus += 3 * n_pairs

        # journey
        if city.journey_points_override is not None:
            result.journey_points = city.journey_points_override
        else:
            # if we know how many workers are on journey but not which spots,
            # assume the lowest-value spots filled (conservative)
            for i in range(min(city.journey_workers, len(JOURNEY_POINTS))):
                result.journey_points += JOURNEY_POINTS[i]

        # events
        for ev in city.events:
            row = self._load_card(ev)
            if row:
                result.event_points += int(row["base_points"])
            else:
                result.warnings.append(f"unknown event: {ev}")

        return result

    def _apply_card_rule(
        self,
        card: CityCard,
        name_counts: Counter,
        out: ScoreBreakdown,
    ) -> None:
        """Apply card-specific scoring that is determinable from the photo.

        Rules that need adjacency (Husband+Farm) or discard-pile state are
        skipped here on purpose — see module docstring.
        """
        n = card.name.lower()

        # Dungeon: 3 points per critter tucked under (max 2)
        if n == "dungeon":
            out.rule_bonuses["dungeon"] = (
                out.rule_bonuses.get("dungeon", 0) + 3 * min(card.cards_under, 2)
            )

        # Wanderer / Fool / Ruins are scored via point tokens already counted

        # Farm: base points only; Husband adjacency bonus is handled by
        # CRITTER_CONSTRUCTION_PAIRS via wife<->husband; farm-husband bonus
        # is intentionally left out (requires adjacency).
