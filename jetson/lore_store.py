"""
LORE Persistence — SQLite store for Qualia's accumulated world-knowledge.

Watches Qualia's SHM LoreBuffer for new entries and persists them to SQLite.
On startup, loads persisted LORE back so knowledge survives restarts.

Phase 7.3: LORE effectiveness feedback loop:
  - Track pre/post VFE around LORE injection to measure actual impact
  - Score LORE entries by effectiveness (did VFE decrease after injection?)
  - Prioritize high-effectiveness entries for LLM context injection
  - Prune low-effectiveness entries after retention period
  - Provide ranked LORE for voice assistant integration

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

DEFAULT_DB_PATH = os.path.expanduser("~/training-data/lore.db")

# Phase 7.3: Effectiveness tracking parameters
VFE_MEASURE_DELAY = 5.0      # seconds after LORE injection to measure VFE
EFFECTIVENESS_EMA = 0.3      # blend factor for updating effectiveness score
RETENTION_DAYS = 30           # prune entries older than this with low effectiveness
MIN_EFFECTIVENESS = 0.1       # threshold below which old entries get pruned
MAX_LLM_CONTEXT_ENTRIES = 15  # max entries for LLM context (ranked by effectiveness)


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
    """Persistent LORE storage with effectiveness tracking (Phase 7.3)."""

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
                pre_vfe REAL DEFAULT 0,
                post_vfe REAL DEFAULT 0,
                vfe_impact REAL DEFAULT 0,
                times_injected INTEGER DEFAULT 0,
                last_injected_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_lore_layer ON lore(layer);
            CREATE INDEX IF NOT EXISTS idx_lore_created ON lore(created_at);
            CREATE INDEX IF NOT EXISTS idx_lore_seq ON lore(shm_seq);
            CREATE INDEX IF NOT EXISTS idx_lore_effectiveness ON lore(effectiveness DESC);
        """)
        # Add columns if they don't exist (migration for existing DBs)
        for col, typ, default in [
            ("pre_vfe", "REAL", "0"),
            ("post_vfe", "REAL", "0"),
            ("vfe_impact", "REAL", "0"),
            ("times_injected", "INTEGER", "0"),
            ("last_injected_at", "TEXT", "NULL"),
        ]:
            try:
                self.conn.execute(
                    f"ALTER TABLE lore ADD COLUMN {col} {typ} DEFAULT {default}"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
        self.conn.commit()

    def save_entry(self, question: str, answer: str, layer: int, reason: int,
                   embedding_delta: float = 0.0, effectiveness: float = 0.0,
                   shm_seq: int = 0, pre_vfe: float = 0.0) -> bool:
        """Save a LORE entry. Returns False if duplicate (same seq)."""
        with self._lock:
            if shm_seq > 0:
                existing = self.conn.execute(
                    "SELECT 1 FROM lore WHERE shm_seq = ?", (shm_seq,)
                ).fetchone()
                if existing:
                    return False

            self.conn.execute(
                "INSERT INTO lore (question, answer, layer, reason, "
                "embedding_delta, effectiveness, shm_seq, pre_vfe) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (question, answer, layer, reason,
                 embedding_delta, effectiveness, shm_seq, pre_vfe),
            )
            self.conn.commit()
            return True

    def update_effectiveness(self, shm_seq: int, post_vfe: float):
        """Phase 7.3: Update effectiveness after measuring post-injection VFE.
        vfe_impact = (pre_vfe - post_vfe) / max(pre_vfe, 0.001)
        Positive impact = VFE decreased = LORE helped."""
        with self._lock:
            row = self.conn.execute(
                "SELECT pre_vfe, effectiveness FROM lore WHERE shm_seq = ?",
                (shm_seq,)
            ).fetchone()
            if not row:
                return

            pre_vfe = row[0]
            old_effectiveness = row[1]
            if pre_vfe > 0.001:
                vfe_impact = (pre_vfe - post_vfe) / pre_vfe
            else:
                vfe_impact = 0.0

            # EMA blend with existing effectiveness score
            new_effectiveness = (
                old_effectiveness * (1 - EFFECTIVENESS_EMA)
                + max(min(vfe_impact, 1.0), -1.0) * EFFECTIVENESS_EMA
            )

            self.conn.execute(
                "UPDATE lore SET post_vfe = ?, vfe_impact = ?, effectiveness = ? "
                "WHERE shm_seq = ?",
                (post_vfe, vfe_impact, new_effectiveness, shm_seq),
            )
            self.conn.commit()

    def record_injection(self, lore_id: int):
        """Phase 7.3: Record that a LORE entry was injected into LLM context."""
        with self._lock:
            self.conn.execute(
                "UPDATE lore SET times_injected = times_injected + 1, "
                "last_injected_at = datetime('now') WHERE id = ?",
                (lore_id,),
            )
            self.conn.commit()

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

    def get_top_effective(self, limit: int = MAX_LLM_CONTEXT_ENTRIES) -> list:
        """Phase 7.3: Get highest-effectiveness LORE entries for LLM context."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, question, answer, layer, reason, effectiveness, "
                "vfe_impact, times_injected, created_at FROM lore "
                "WHERE effectiveness > 0 "
                "ORDER BY effectiveness DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0], "question": r[1], "answer": r[2], "layer": r[3],
                "reason": r[4], "effectiveness": r[5], "vfe_impact": r[6],
                "times_injected": r[7], "created_at": r[8],
            }
            for r in rows
        ]

    def search(self, query: str, limit: int = 10) -> list:
        """Search LORE by keyword in question or answer."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT question, answer, layer, reason, effectiveness, created_at "
                "FROM lore "
                "WHERE question LIKE ? OR answer LIKE ? "
                "ORDER BY effectiveness DESC, created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [
            {"question": r[0], "answer": r[1], "layer": r[2],
             "reason": r[3], "effectiveness": r[4], "created_at": r[5]}
            for r in rows
        ]

    def get_for_llm_context(self, limit: int = MAX_LLM_CONTEXT_ENTRIES) -> str:
        """Phase 7.3: Get LORE ranked by effectiveness for LLM context injection.
        High-effectiveness entries first (proven to reduce VFE)."""
        entries = self.get_top_effective(limit)
        recent = self.get_recent(5)  # also include most recent regardless of score

        # Merge: top effective + recent, deduplicate
        seen_questions = set()
        merged = []
        for e in entries + recent:
            q = e["question"][:80]
            if q not in seen_questions:
                seen_questions.add(q)
                merged.append(e)
                if len(merged) >= limit:
                    break

        if not merged:
            return ""

        # Record injection
        for e in merged:
            if "id" in e:
                self.record_injection(e["id"])

        lines = ["[Accumulated World Knowledge (LORE) — ranked by effectiveness:]"]
        for e in merged:
            eff = e.get("effectiveness", 0)
            eff_str = f" [eff={eff:.2f}]" if eff else ""
            lines.append(f"- Q: {e['question'][:100]}{eff_str}")
            lines.append(f"  A: {e['answer'][:150]}")
        return "\n".join(lines)

    def prune_ineffective(self) -> int:
        """Phase 7.3: Remove old entries with low effectiveness.
        Keeps DB from growing unboundedly."""
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM lore WHERE effectiveness < ? "
                "AND created_at < datetime('now', ?)",
                (MIN_EFFECTIVENESS, f"-{RETENTION_DAYS} days"),
            )
            count = cursor.rowcount
            if count > 0:
                self.conn.commit()
                log.info(f"Pruned {count} ineffective LORE entries "
                         f"(older than {RETENTION_DAYS}d, eff < {MIN_EFFECTIVENESS})")
            return count

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
            avg_eff = self.conn.execute(
                "SELECT AVG(effectiveness) FROM lore WHERE effectiveness != 0"
            ).fetchone()[0] or 0.0
            effective_count = self.conn.execute(
                "SELECT COUNT(*) FROM lore WHERE effectiveness > 0.1"
            ).fetchone()[0]
        return {
            "total": total,
            "by_layer": {r[0]: r[1] for r in by_layer},
            "by_reason": {r[0]: r[1] for r in by_reason},
            "avg_embedding_delta": round(avg_delta, 4),
            "avg_effectiveness": round(avg_eff, 4),
            "effective_entries": effective_count,
            "effectiveness_rate": round(effective_count / max(total, 1), 3),
        }

    def close(self):
        self.conn.close()


# ── SHM Watcher ──────────────────────────────────────────────────

def watch_lore_buffer(store: LoreStore, poll_hz: float = 2.0):
    """Watch Qualia SHM for new LORE entries and persist them.
    Phase 7.3: Also tracks VFE before/after injection for effectiveness scoring."""
    try:
        from qualia_bridge import QualiaBridge, NUM_LAYERS
    except ImportError:
        log.error("qualia_bridge module not found")
        return

    bridge = QualiaBridge()
    if not bridge.open():
        log.error("Cannot open Qualia SHM — is qualia-watch running?")
        return

    log.info("Watching LORE buffer at %.1f Hz (Phase 7.3: effectiveness tracking)", poll_hz)

    last_seq = 0
    tick = 1.0 / poll_hz

    # Phase 7.3: Pending effectiveness measurements
    # Maps shm_seq → (injection_time, pre_vfe)
    pending_measurements: dict = {}

    prune_interval = 3600  # prune every hour
    last_prune = time.monotonic()

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while running:
        try:
            entries = bridge.read_recent_lore(20)
            new_count = 0

            # Measure current mean VFE across all layers for pre-injection baseline
            try:
                layer_vfes = [bridge.read_layer_belief(i).vfe for i in range(NUM_LAYERS)]
                current_mean_vfe = sum(layer_vfes) / len(layer_vfes) if layer_vfes else 0.0
            except Exception:
                current_mean_vfe = 0.0

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
                        pre_vfe=current_mean_vfe,
                    )
                    if saved:
                        new_count += 1
                        # Schedule effectiveness measurement
                        pending_measurements[entry.seq] = (
                            time.monotonic(), current_mean_vfe
                        )
                        log.info(
                            "Persisted LORE #%d L%d (pre_vfe=%.4f): %s → %s",
                            entry.seq, entry.layer, current_mean_vfe,
                            entry.question[:40], entry.answer[:40],
                        )
                    last_seq = max(last_seq, entry.seq)

            if new_count > 0:
                log.info("Persisted %d new entries (total: %d)", new_count, store.count())

            # Phase 7.3: Check pending effectiveness measurements
            now = time.monotonic()
            completed = []
            for seq, (inject_time, pre_vfe) in pending_measurements.items():
                if now - inject_time >= VFE_MEASURE_DELAY:
                    # Measure post-injection VFE
                    try:
                        post_vfes = [
                            bridge.read_layer_belief(i).vfe
                            for i in range(NUM_LAYERS)
                        ]
                        post_mean_vfe = sum(post_vfes) / len(post_vfes)
                    except Exception:
                        post_mean_vfe = pre_vfe  # no change if can't read

                    store.update_effectiveness(seq, post_mean_vfe)
                    impact = (pre_vfe - post_mean_vfe) / max(pre_vfe, 0.001)
                    log.info(
                        "LORE #%d effectiveness: pre=%.4f post=%.4f impact=%.1f%%",
                        seq, pre_vfe, post_mean_vfe, impact * 100,
                    )
                    completed.append(seq)

            for seq in completed:
                del pending_measurements[seq]

            # Periodic pruning of ineffective entries
            if now - last_prune > prune_interval:
                store.prune_ineffective()
                last_prune = now

        except Exception as e:
            log.error("LORE watch error: %s", e)

        time.sleep(tick)

    bridge.close()
    store.close()
    log.info("LORE store shutdown")


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
    parser.add_argument("--effective", action="store_true",
                        help="Show top effective LORE entries")
    args = parser.parse_args()

    store = LoreStore(args.db)

    if args.dump:
        entries = store.get_recent(100)
        if not entries:
            print("No LORE entries yet.")
        for e in entries:
            print(f"[{e['created_at']}] L{e['layer']} r={e['reason']} "
                  f"eff={e['effectiveness']:.3f}:")
            print(f"  Q: {e['question']}")
            print(f"  A: {e['answer']}")
            print()
    elif args.stats:
        s = store.stats()
        print(f"Total LORE entries: {s['total']}")
        print(f"By layer: {s['by_layer']}")
        print(f"By reason: {s['by_reason']}")
        print(f"Avg embedding delta: {s['avg_embedding_delta']}")
        print(f"Avg effectiveness: {s['avg_effectiveness']}")
        print(f"Effective entries (>0.1): {s['effective_entries']} "
              f"({s['effectiveness_rate']:.0%})")
    elif args.search:
        results = store.search(args.search)
        for r in results:
            print(f"L{r['layer']} [eff={r['effectiveness']:.2f}]: "
                  f"Q: {r['question'][:80]}")
            print(f"       A: {r['answer'][:100]}")
            print()
    elif args.effective:
        entries = store.get_top_effective(20)
        if not entries:
            print("No effective LORE entries yet.")
        for e in entries:
            print(f"[eff={e['effectiveness']:.3f} impact={e['vfe_impact']:.3f} "
                  f"injected={e['times_injected']}x] L{e['layer']}:")
            print(f"  Q: {e['question'][:80]}")
            print(f"  A: {e['answer'][:100]}")
            print()
    else:
        # Run as daemon
        acquire_singleton("lore")
        watch_lore_buffer(store)
