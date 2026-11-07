"""Append-only memory snapshots with checked, atomic activation."""

import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from shrek.schemas import Proposal, Rule, Snapshot, nonempty


class StaleProposalError(RuntimeError):
    pass


def encode_rules(rules: tuple[Rule, ...]) -> str:
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("Duplicate rule IDs")
    return json.dumps([rule.to_dict() for rule in rules], sort_keys=True)


class MemoryStore:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS versions (
                version INTEGER PRIMARY KEY, parent INTEGER, rules TEXT NOT NULL,
                digest TEXT NOT NULL, reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS proposals (
                change_id TEXT PRIMARY KEY, base_version INTEGER NOT NULL, operation TEXT NOT NULL,
                rules TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
                decision TEXT);
        """)
        self.connection.execute("INSERT OR IGNORE INTO versions VALUES(0,NULL,?,?,?)", ("[]", hashlib.sha256(b"[]").hexdigest(), "initial empty memory"))
        self.connection.execute("INSERT OR IGNORE INTO state VALUES('active',0)")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def snapshot(self, version: int | None = None) -> Snapshot:
        if version is None:
            version = self.connection.execute("SELECT value FROM state WHERE key='active'").fetchone()[0]
        row = self.connection.execute("SELECT rules,digest FROM versions WHERE version=?", (version,)).fetchone()
        if row is None:
            raise ValueError("Unknown memory version")
        return Snapshot(version, tuple(Rule.from_dict(item) for item in json.loads(row[0])), row[1])

    def propose(self, base: Snapshot, operation: str, rules: tuple[Rule, ...], reason: str) -> Proposal:
        if operation not in {"ADD", "REVISE", "ARCHIVE", "REPLACE", "RESTORE"}:
            raise ValueError("Unknown memory operation")
        actual = self.snapshot(base.version)
        if actual != base:
            raise ValueError("Snapshot does not match persistent state")
        payload = encode_rules(rules)
        change = Proposal(uuid.uuid4().hex, operation, base.version, rules, nonempty(reason, "reason"))
        self.connection.execute("INSERT INTO proposals VALUES(?,?,?,?,?,'pending',NULL)", (change.change_id, change.base_version, operation, payload, change.reason))
        self.connection.commit()
        return change

    def reject(self, change: Proposal, reason: str) -> None:
        cursor = self.connection.execute("UPDATE proposals SET status='rejected',decision=? WHERE change_id=? AND status='pending'", (nonempty(reason, "rejection reason"), change.change_id))
        if cursor.rowcount != 1:
            self.connection.rollback()
            raise ValueError("Proposal is missing or already decided")
        self.connection.commit()

    def commit(self, change: Proposal, evidence: dict) -> Snapshot:
        """Caller supplies a decision; this store does not implement the SHREK gate."""
        encoded_evidence = json.dumps(evidence, sort_keys=True)
        payload = encode_rules(change.candidate_rules)
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute("SELECT base_version,operation,rules,reason,status FROM proposals WHERE change_id=?", (change.change_id,)).fetchone()
            if row != (change.base_version, change.operation, payload, change.reason, "pending"):
                raise ValueError("Proposal differs from stored pending proposal")
            active = self.connection.execute("SELECT value FROM state WHERE key='active'").fetchone()[0]
            if active != change.base_version:
                raise StaleProposalError("Memory changed since proposal creation")
            version = self.connection.execute("SELECT MAX(version)+1 FROM versions").fetchone()[0]
            digest = hashlib.sha256(payload.encode()).hexdigest()
            self.connection.execute("INSERT INTO versions VALUES(?,?,?,?,?)", (version, active, payload, digest, change.reason))
            self.connection.execute("UPDATE state SET value=? WHERE key='active'", (version,))
            self.connection.execute("UPDATE proposals SET status='accepted',decision=? WHERE change_id=?", (encoded_evidence, change.change_id))
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return self.snapshot(version)

    def resolve(self, change: Proposal, status: str, evidence: dict) -> None:
        """Persist a rejection/deferral without activating any part of a candidate."""
        if status not in {"rejected", "deferred"}:
            raise ValueError("Only commit can accept a proposal")
        cursor = self.connection.execute("UPDATE proposals SET status=?,decision=? WHERE change_id=? AND status='pending'",
                                         (status, json.dumps(evidence, sort_keys=True), change.change_id))
        if cursor.rowcount != 1:
            self.connection.rollback()
            raise ValueError("Proposal is missing or already decided")
        self.connection.commit()

    def restore(self, historical_version: int, reason: str) -> Snapshot:
        """Explicit operator rollback, logged as a new version rather than rewriting history."""
        change = self.propose(self.snapshot(), "RESTORE", self.snapshot(historical_version).rules, reason)
        return self.commit(change, {"decision_source": "explicit_operator_rollback", "restored_version": historical_version})
