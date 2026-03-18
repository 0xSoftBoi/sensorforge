"""
LORE Persistence — SQLite store for Qualia's accumulated world-knowledge.

Watches Qualia's SHM LoreBuffer for new entries and persists them to SQLite.
On startup, loads persisted LORE back so knowledge survives restarts.

Usage:
    python3 lore_store.py                # Run as daemon
    python3 lore_store.py --dump         # Print all stored LORE
    python3 lore_store.py --stats        # Show LORE statistics
"""

import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from typing import Optional

log = logging.getLogger("lore-store")

# Default DB path (same as voice assistant's conversations DB dir)
DEFAULT_DB_PATH = os.path.expanduser("~/training-data/lore.db")


def acquire_singleton(name):
    """Ensure only one instance runs. Kill stale instance if found."""
    pidfile = f"/tmp/qualia_{name}.pid"
    try:
        with open(pidfile) as f:
            old_pid = int(f.read().strip())
        os.kill(old_pid, 0)
        log.warning(f"Killing stale {name} process (PID {old_pid})")
        os.kill(old_pid, signal.SIGTERM)
        time.sleep(1)
        try:
            os.kill(old_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    log.info(f"Singleton lock acquired: {pidfile} (PID {os.getpid()})")


class LoreStore:
    """Persistent LORE storage using SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS lore (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                layer INTEGER NOT NULL,
                reason INTEGER NOT NULL,
                embedding_delta REAL DEFAULT 0,
                effectiveness REAL DEFAULT 0,
                shm_seq INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_lore_layer ON lore(layer);
            CREATE INDEX IF NOT EXISTS idx_lore_created ON lore(created_at);
            CREATE INDEX IF NOT EXISTS idx_lore_seq ON lore(shm_seq);
        """)
        self.conn.commit()

    def save_entry(self, question: str, answer: str, layer: int, reason: int,
                   embedding_delta: float = 0.0, effectiveness: float = 0.0,
                   shm_seq: int = 0) -> bool:
        """Save a LORE entry. Returns False if duplicate (same seq)."""
        with self._lock:
            # Skip duplicates by SHM sequence number
            if shm_seq > 0:
                existing = self.conn.execute(
                    "SELECT 1 FROM lore WHERE shm_seq = ?", (shm_seq,)
                ).fetchone()
                if existing:
                    return False

            self.conn.execute(
                "INSERT INTO lore (question, answer, layer, reason, "
                "embedding_delta, effectiveness, shm_seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (question, answer, layer, reason,
                 embedding_delta, effectiveness, shm_seq),
            )
            self.conn.commit()
            return True

    def get_recent(self, limit: int = 20) -> list:
        """Get most recent LORE entries."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT question, answer, layer, reason, embedding_delta, "
                "effectiveness, created_at FROM lore "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "question": r[0], "answer": r[1], "layer": r[2],
                "reason": r[3], "embedding_delta": r[4],
                "effectiveness": r[5], "created_at": r[6],
            }
            for r in reversed(rows)
        ]

    def search(self, query: str, limit: int = 10) -> list:
        """Search LORE by keyword in question or answer."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT question, answer, layer, reason, created_at FROM lore "
                "WHERE question LIKE ? OR answer LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [
            {"question": r[0], "answer": r[1], "layer": r[2],
             "reason": r[3], "created_at": r[4]}
            for r in rows
        ]

    def get_for_llm_context(self, limit: int = 10) -> str:
        """Get LORE formatted for injection into LLM conversation context."""
        entries = self.get_recent(limit)
        if not entries:
            return ""
        lines = ["[Accumulated World Knowledge (LORE):]"]
        for e in entries:
            lines.append(f"- Q: {e['question'][:100]}")
            lines.append(f"  A: {e['answer'][:150]}")
        return "\n".join(lines)

    def count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM lore").fetchone()[0]

    def stats(self) -> dict:
        with self._lock:
            total = self.conn.execute("SELECT COUNT(*) FROM lore").fetchone()[0]
            by_layer = self.conn.execute(
                "SELECT layer, COUNT(*) FROM lore GROUP BY layer ORDER BY layer"
            ).fetchall()
            by_reason = self.conn.execute(
                "SELECT reason, COUNT(*) FROM lore GROUP BY reason ORDER BY reason"
            ).fetchall()
            avg_delta = self.conn.execute(
                "SELECT AVG(embedding_delta) FROM lore"
            ).fetchone()[0] or 0.0
        return {
            "total": total,
            "by_layer": {r[0]: r[1] for r in by_layer},
            "by_reason": {r[0]: r[1] for r in by_reason},
            "avg_embedding_delta": round(avg_delta, 4),
        }

    def close(self):
        self.conn.close()


# ── SHM Watcher ──────────────────────────────────────────────────

def watch_lore_buffer(store: LoreStore, poll_hz: float = 2.0):
    """Watch Qualia SHM for new LORE entries and persist them."""
    try:
        from qualia_bridge import QualiaBridge
    except ImportError:
        log.error("qualia_bridge module not found")
        return

    bridge = QualiaBridge()
    if not bridge.open():
        log.error("Cannot open Qualia SHM — is qualia-watch running?")
        return

    log.info("Watching LORE buffer at %.1f Hz", poll_hz)

    last_seq = 0
    tick = 1.0 / poll_hz

    while True:
        try:
            entries = bridge.read_recent_lore(20)
            new_count = 0

            for entry in entries:
                if entry.seq > last_seq:
                    saved = store.save_entry(
                        question=entry.question,
                        answer=entry.answer,
                        layer=entry.layer,
                        reason=entry.reason,
                        embedding_delta=entry.embedding_delta,
                        effectiveness=entry.effectiveness,
                        shm_seq=entry.seq,
                    )
                    if saved:
                        new_count += 1
                        log.info(
                            "Persisted LORE #%d L%d: %s → %s",
                            entry.seq, entry.layer,
                            entry.question[:40], entry.answer[:40],
                        )
                    last_seq = max(last_seq, entry.seq)

            if new_count > 0:
                log.info("Persisted %d new LORE entries (total: %d)", new_count, store.count())

        except Exception as e:
            log.error("LORE watch error: %s", e)

        time.sleep(tick)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="LORE Persistence Store")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument("--dump", action="store_true", help="Dump all LORE entries")
    parser.add_argument("--stats", action="store_true", help="Show LORE statistics")
    parser.add_argument("--search", type=str, help="Search LORE by keyword")
    args = parser.parse_args()

    store = LoreStore(args.db)

    if args.dump:
        entries = store.get_recent(100)
        if not entries:
            print("No LORE entries yet.")
        for e in entries:
            print(f"[{e['created_at']}] L{e['layer']} r={e['reason']}:")
            print(f"  Q: {e['question']}")
            print(f"  A: {e['answer']}")
            print()
    elif args.stats:
        s = store.stats()
        print(f"Total LORE entries: {s['total']}")
        print(f"By layer: {s['by_layer']}")
        print(f"By reason: {s['by_reason']}")
        print(f"Avg embedding delta: {s['avg_embedding_delta']}")
    elif args.search:
        results = store.search(args.search)
        for r in results:
            print(f"L{r['layer']}: Q: {r['question'][:80]}")
            print(f"       A: {r['answer'][:100]}")
            print()
    else:
        # Run as daemon — watch SHM and persist LORE
        acquire_singleton("lore")
        watch_lore_buffer(store)
