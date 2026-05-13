import sqlite3
from typing import Optional


class EverdellDB:
    def __init__(self, db_path: str = "everdell_cards.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS card_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                rarity TEXT CHECK (rarity IN ('unique', 'common')),
                card_type TEXT CHECK (card_type IN ('critter', 'construction')),
                color TEXT,
                expansion TEXT,
                base_points INTEGER DEFAULT 0,
                scoring_rule TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                reference TEXT NOT NULL,
                FOREIGN KEY (name) REFERENCES card_data(name)
            )
        """)

        # additive migration for pre-existing DBs
        cur.execute("PRAGMA table_info(card_data)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "base_points" not in existing_cols:
            cur.execute("ALTER TABLE card_data ADD COLUMN base_points INTEGER DEFAULT 0")
        if "scoring_rule" not in existing_cols:
            cur.execute("ALTER TABLE card_data ADD COLUMN scoring_rule TEXT DEFAULT ''")

        conn.commit()
        conn.close()

    def add_card(
        self,
        name: str,
        description: str,
        rarity: str,
        card_type: str,
        color: str,
        expansion: str,
        base_points: int = 0,
        scoring_rule: str = "",
        notes: str = ""
    ):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO card_data (name, description, rarity, card_type, color, expansion, base_points, scoring_rule, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, description, rarity, card_type, color, expansion, base_points, scoring_rule, notes))
        conn.commit()
        conn.close()

    def add_image(self, name: str, reference: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO images (name, reference)
            VALUES (?, ?)
        """, (name, reference))
        conn.commit()
        conn.close()

    def get_card(self, name: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM card_data WHERE name = ?", (name,))
        row = cur.fetchone()
        conn.close()
        return row

    def get_images_for_card(self, name: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM images WHERE name = ?", (name,))
        rows = cur.fetchall()
        conn.close()
        return rows


if __name__ == "__main__":
    db = EverdellDB()
    print("Database ready.")