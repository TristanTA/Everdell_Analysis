import csv
import sqlite3
from pathlib import Path

from init_db import EverdellDB


CSV_PATH = Path("data/card_data.csv")
DB_PATH = "everdell_cards.db"


def normalize(value):
    if value is None:
        return ""
    return str(value).strip()


def row_from_csv(csv_row):
    return {
        "name": normalize(csv_row.get("name")),
        "description": normalize(csv_row.get("description")),
        "rarity": normalize(csv_row.get("rarity")).lower(),
        "card_type": normalize(csv_row.get("card_type")).lower(),
        "color": normalize(csv_row.get("color")).lower(),
        "expansion": normalize(csv_row.get("expansion")).lower(),
        "notes": normalize(csv_row.get("notes")),
    }


def get_db_cards(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT name, description, rarity, card_type, color, expansion, notes
        FROM card_data
    """)
    rows = cur.fetchall()

    cards = {}
    for row in rows:
        cards[normalize(row[0])] = {
            "name": normalize(row[0]),
            "description": normalize(row[1]),
            "rarity": normalize(row[2]).lower(),
            "card_type": normalize(row[3]).lower(),
            "color": normalize(row[4]).lower(),
            "expansion": normalize(row[5]).lower(),
            "notes": normalize(row[6]),
        }
    return cards


def sync_card_data(csv_path=CSV_PATH, db_path=DB_PATH):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # ensures db + tables exist
    EverdellDB(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    db_cards = get_db_cards(conn)

    inserted = 0
    updated = 0
    skipped = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"name", "description", "rarity", "card_type", "color", "expansion", "notes"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for raw_row in reader:
            card = row_from_csv(raw_row)

            if not card["name"]:
                continue

            existing = db_cards.get(card["name"])

            if existing is None:
                cur.execute("""
                    INSERT INTO card_data (
                        name, description, rarity, card_type, color, expansion, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    card["name"],
                    card["description"],
                    card["rarity"],
                    card["card_type"],
                    card["color"],
                    card["expansion"],
                    card["notes"],
                ))
                inserted += 1
                continue

            fields_changed = any(
                existing[key] != card[key]
                for key in ["description", "rarity", "card_type", "color", "expansion", "notes"]
            )

            if fields_changed:
                cur.execute("""
                    UPDATE card_data
                    SET description = ?,
                        rarity = ?,
                        card_type = ?,
                        color = ?,
                        expansion = ?,
                        notes = ?
                    WHERE name = ?
                """, (
                    card["description"],
                    card["rarity"],
                    card["card_type"],
                    card["color"],
                    card["expansion"],
                    card["notes"],
                    card["name"],
                ))
                updated += 1
            else:
                skipped += 1

    conn.commit()
    conn.close()

    print(f"Sync complete.")
    print(f"Inserted: {inserted}")
    print(f"Updated:  {updated}")
    print(f"Unchanged:{skipped}")


if __name__ == "__main__":
    sync_card_data()