"""Durable, tenant-scoped shadow jobs; callers explicitly drive all transitions.

No background worker or default runtime path is created. Lease mutations return
None when fenced, disabled, missing, or terminal. A failed attempt is retriable;
its checkpoint remains available to the next lease holder.
"""

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager

TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"leased", "running"}


class ResearchShadowStore:
    def __init__(self, path, clock=time.time):
        self.path = str(path)
        self.clock = clock
        with self._transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS shadow_jobs (
                    tenant_id TEXT NOT NULL, id TEXT NOT NULL,
                    assignment_id TEXT NOT NULL, source_hash TEXT NOT NULL,
                    policy_version TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id),
                    UNIQUE (tenant_id, assignment_id, source_hash, policy_version)
                );
                CREATE TABLE IF NOT EXISTS shadow_controls (
                    tenant_id TEXT NOT NULL, assignment_id TEXT NOT NULL,
                    disabled INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, assignment_id)
                );
            """)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _scope(tenant_id):
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required")

    @staticmethod
    def _budget(config, key, default):
        value = config.get("budgets", {}).get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid budget: {key}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid budget: {key}")
        if key != "max_elapsed_seconds" and int(value) != value:
            raise ValueError(f"Invalid integer budget: {key}")
        return value

    @staticmethod
    def _read(db, tenant_id, job_id):
        row = db.execute(
            "SELECT data FROM shadow_jobs WHERE tenant_id=? AND id=?",
            (tenant_id, job_id),
        ).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _save(db, job):
        db.execute(
            "UPDATE shadow_jobs SET data=? WHERE tenant_id=? AND id=?",
            (json.dumps(job, allow_nan=False), job["tenant_id"], job["id"]),
        )
        return job

    @staticmethod
    def _disabled(db, job):
        return (
            db.execute(
                "SELECT 1 FROM shadow_controls WHERE tenant_id=? "
                "AND assignment_id IN ('', ?) AND disabled=1 LIMIT 1",
                (job["tenant_id"], job["assignment_id"]),
            ).fetchone()
            is not None
        )

    def enqueue(self, tenant_id, assignment_id, source_hash, policy_version, config):
        self._scope(tenant_id)
        if not all(
            isinstance(v, str) and v.strip() for v in (assignment_id, source_hash, policy_version)
        ):
            raise ValueError("Assignment, source hash and policy version are required")
        if not isinstance(config, dict) or not isinstance(config.get("budgets", {}), dict):
            raise ValueError("Configuration and budgets must be dictionaries")
        if "tenant_id" in config and config["tenant_id"] != tenant_id:
            raise ValueError("Configuration tenant_id does not match job tenant")
        config = json.loads(json.dumps(config, allow_nan=False))
        for key, default in (
            ("max_concurrency", 1),
            ("max_attempts", 3),
            ("max_elapsed_seconds", 60),
        ):
            self._budget(config, key, default)
        with self._transaction() as db:
            row = db.execute(
                "SELECT data FROM shadow_jobs WHERE tenant_id=? AND assignment_id=? "
                "AND source_hash=? AND policy_version=?",
                (tenant_id, assignment_id, source_hash, policy_version),
            ).fetchone()
            if row:
                existing = json.loads(row[0])
                if existing["config"] != config:
                    raise ValueError("Idempotency key already has a different configuration")
                return existing
            job = dict(
                id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                assignment_id=assignment_id,
                source_hash=source_hash,
                policy_version=policy_version,
                config=config,
                state="queued",
                attempts=0,
                lease_token=None,
                lease_owner=None,
                lease_expires_at=None,
                checkpoint=None,
                receipt=None,
                reason=None,
                created_at=self.clock(),
                first_started_at=None,
                available_at=0,
            )
            db.execute(
                "INSERT INTO shadow_jobs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tenant_id,
                    job["id"],
                    assignment_id,
                    source_hash,
                    policy_version,
                    json.dumps(job, allow_nan=False),
                ),
            )
            return job

    def _expired_budget(self, job, now):
        started = job["first_started_at"]
        return started is not None and now - started >= self._budget(
            job["config"], "max_elapsed_seconds", 60
        )

    @staticmethod
    def _clear_lease(job):
        job.update(lease_token=None, lease_owner=None, lease_expires_at=None)

    def _retry(self, job, now, reason):
        self._clear_lease(job)
        job["reason"] = reason
        if self._expired_budget(job, now):
            job.update(state="failed", reason="budget_exhausted")
        elif job["attempts"] >= self._budget(job["config"], "max_attempts", 3):
            job["state"] = "failed"
        else:
            job.update(state="queued", available_at=now + min(2 ** min(job["attempts"] - 1, 5), 30))

    @staticmethod
    def _lease_duration(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Invalid lease duration")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid lease duration")
        return min(value, 300)

    def claim(self, tenant_id, job_id, owner, lease_seconds=30):
        self._scope(tenant_id)
        duration = self._lease_duration(lease_seconds)
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("owner is required")
        with self._transaction() as db:
            job = self._read(db, tenant_id, job_id)
            now = self.clock()
            if not job or job["state"] in TERMINAL or self._disabled(db, job):
                return None
            if job["state"] in ACTIVE:
                if job["lease_expires_at"] > now:
                    return None
                self._retry(job, now, "lease_expired")
                self._save(db, job)
            if self._expired_budget(job, now):
                job.update(state="failed", reason="budget_exhausted")
                self._clear_lease(job)
                self._save(db, job)
                return None
            if job["state"] != "queued" or job["available_at"] > now:
                return None
            rows = db.execute("SELECT data FROM shadow_jobs WHERE tenant_id=?", (tenant_id,))
            active = [json.loads(row[0]) for row in rows]
            active = [j for j in active if j["state"] in ACTIVE and j["lease_expires_at"] > now]
            # Honor the strictest concurrency bound among competing leases.
            limit = min(
                [self._budget(job["config"], "max_concurrency", 1)]
                + [self._budget(j["config"], "max_concurrency", 1) for j in active]
            )
            if len(active) >= limit:
                return None
            job.update(
                state="leased",
                attempts=job["attempts"] + 1,
                lease_token=uuid.uuid4().hex,
                lease_owner=owner,
                lease_expires_at=now + duration,
            )
            return self._save(db, job)

    def _mutate(self, tenant_id, job_id, token, action, payload=None):
        self._scope(tenant_id)
        with self._transaction() as db:
            job = self._read(db, tenant_id, job_id)
            now = self.clock()
            if (
                not job
                or job["state"] not in ACTIVE
                or not token
                or job["lease_token"] != token
                or self._disabled(db, job)
            ):
                return None
            if self._expired_budget(job, now):
                job.update(state="failed", reason="budget_exhausted")
                self._clear_lease(job)
                self._save(db, job)
                return None
            if job["lease_expires_at"] <= now:
                return None
            if action == "start":
                job["state"] = "running"
                if job["first_started_at"] is None:
                    job["first_started_at"] = now
            elif action == "heartbeat":
                job["lease_expires_at"] = now + payload
            elif action == "checkpoint":
                if job["state"] != "running":
                    return None
                job["checkpoint"] = payload
            elif action == "complete":
                if job["state"] != "running":
                    return None
                job.update(state="completed", receipt=payload, reason=None)
                self._clear_lease(job)
            elif action == "fail":
                self._retry(job, now, payload)
            return self._save(db, job)

    def start(self, tenant_id, job_id, lease_token):
        return self._mutate(tenant_id, job_id, lease_token, "start")

    def heartbeat(self, tenant_id, job_id, lease_token, lease_seconds=30):
        return self._mutate(
            tenant_id, job_id, lease_token, "heartbeat", self._lease_duration(lease_seconds)
        )

    @staticmethod
    def _payload(value):
        if not isinstance(value, dict):
            raise ValueError("Payload must be a dictionary")
        return json.loads(json.dumps(value, allow_nan=False))

    def checkpoint(self, tenant_id, job_id, lease_token, payload):
        return self._mutate(tenant_id, job_id, lease_token, "checkpoint", self._payload(payload))

    def complete(self, tenant_id, job_id, lease_token, receipt):
        return self._mutate(tenant_id, job_id, lease_token, "complete", self._payload(receipt))

    def fail(self, tenant_id, job_id, lease_token, reason="provider_unavailable"):
        return self._mutate(tenant_id, job_id, lease_token, "fail", str(reason))

    def cancel(self, tenant_id, job_id):
        self._scope(tenant_id)
        with self._transaction() as db:
            job = self._read(db, tenant_id, job_id)
            if job and job["state"] not in TERMINAL:
                job.update(state="cancelled", reason="cancelled")
                self._clear_lease(job)
                self._save(db, job)
            return job

    def get(self, tenant_id, job_id):
        self._scope(tenant_id)
        with self._transaction() as db:
            return self._read(db, tenant_id, job_id)

    def list_jobs(self, tenant_id):
        self._scope(tenant_id)
        with self._transaction() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT data FROM shadow_jobs WHERE tenant_id=? ORDER BY rowid", (tenant_id,)
                )
            ]

    def get_controls(self, tenant_id, assignment_id):
        self._scope(tenant_id)
        if not isinstance(assignment_id, str) or not assignment_id.strip():
            raise ValueError("assignment_id is required")
        with self._transaction() as db:
            controls = dict(
                db.execute(
                    "SELECT assignment_id, disabled FROM shadow_controls "
                    "WHERE tenant_id=? AND assignment_id IN ('', ?)",
                    (tenant_id, assignment_id),
                )
            )
            return {
                "paused": bool(controls.get(assignment_id, 0)),
                "killed": bool(controls.get("", 0)),
            }

    def _control(self, tenant_id, assignment_id, disabled):
        self._scope(tenant_id)
        with self._transaction() as db:
            db.execute(
                "INSERT INTO shadow_controls VALUES (?, ?, ?) "
                "ON CONFLICT(tenant_id, assignment_id) DO UPDATE SET disabled=excluded.disabled",
                (tenant_id, assignment_id, int(bool(disabled))),
            )

    def set_paused(self, tenant_id, assignment_id, paused):
        if not isinstance(assignment_id, str) or not assignment_id.strip():
            raise ValueError("assignment_id is required")
        self._control(tenant_id, assignment_id, paused)

    def set_killed(self, tenant_id, killed):
        self._control(tenant_id, "", killed)
