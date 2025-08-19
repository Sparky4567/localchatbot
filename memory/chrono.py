from __future__ import annotations
import sqlite3
import time
from typing import List, Tuple

class ChronoMemory:
    def __init__(self, db_path: str = "chat.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                """                
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            con.commit()

    def add_message(self, role: str, text: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO messages (role, text, ts) VALUES (?, ?, ?)",
                (role, text, time.time()),
            )
            con.commit()

    def last_n(self, n: int = 6) -> List[Tuple[str, str]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT role, text FROM messages ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        rows.reverse()
        return [(r, t) for (r, t) in rows]
