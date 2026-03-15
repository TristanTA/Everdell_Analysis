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
        notes: str = ""
    ):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO card_data (name, description, rarity, card_type, color, expansion, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, description, rarity, card_type, color, expansion, notes))
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