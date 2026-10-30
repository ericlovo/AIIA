"""Durable research-only conductor. Importing this module starts no work.

Every attempt reserves its full cap, including abandoned attempts. Reservations are
not refunded: this deliberately overcounts rather than letting crashes hide spend.
The runtime defaults to replay and never sends outreach or writes to a CRM.
"""

import hashlib
import ipaddress
import json
import math
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .bd_cadence import RESEARCH_SEGMENTS, around_the_clock, validate_cadence

CONFIG_PATH = Path(__file__).with_name("config") / "bd_roster.v1.json"
ACTIVE = {"leased", "running"}
TERMINAL = {"completed", "failed", "cancelled"}
RESOURCES = ("tokens", "calls", "seconds")


class LeaseLost(RuntimeError):
    pass


class BDConductor:
    def __init__(self, db_path, config_path=None, *, config=None, clock=time.time):
        self.path = str(db_path)
        self.clock = clock
        self.config = (
            json.loads(json.dumps(config))
            if config is not None
            else json.loads(Path(config_path or CONFIG_PATH).read_text())
        )
        self._validate_config()
        self.bots = {bot["id"]: bot for bot in self.config["bots"]}
        with self._transaction() as db:
            db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS bd_jobs (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, entity_id TEXT NOT NULL,
                    task TEXT NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL,
                    UNIQUE(tenant_id, entity_id, task)
                );
                CREATE TABLE IF NOT EXISTS bd_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                    kind TEXT NOT NULL, job_id TEXT, data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bd_controls (
                    key TEXT PRIMARY KEY, value INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bd_budget (
                    day TEXT NOT NULL, scope TEXT NOT NULL, resource TEXT NOT NULL,
                    reserved REAL NOT NULL, PRIMARY KEY(day, scope, resource)
                );
                CREATE TABLE IF NOT EXISTS bd_steering (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bd_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                    action TEXT NOT NULL, data TEXT NOT NULL
                );
            """)

    def _validate_config(self):
        cfg = self.config
        if cfg.get("version") != 1 or cfg.get("mode") not in {"replay", "live"}:
            raise ValueError("Version 1 and explicit replay or live mode are required")
        if not isinstance(cfg.get("schedule_enabled"), bool):
            raise ValueError("schedule_enabled must be boolean")
        if "cadence" in cfg:
            validate_cadence(cfg["cadence"])
        if "schedule_window" in cfg:
            window = cfg["schedule_window"]
            if not isinstance(window, dict) or set(window) != {"start_at", "end_at", "timezone"}:
                raise ValueError("schedule_window requires start_at, end_at and timezone")
            try:
                if not all(isinstance(value, str) and value.strip() for value in window.values()):
                    raise ValueError("Schedule window values must be strings")
                start, end = (
                    datetime.fromisoformat(window[key].replace("Z", "+00:00"))
                    for key in ("start_at", "end_at")
                )
                if start.utcoffset() is None or end.utcoffset() is None or end <= start:
                    raise ValueError("Schedule window must be timezone-aware with end after start")
                ZoneInfo(window["timezone"])
            except (ValueError, ZoneInfoNotFoundError) as exc:
                raise ValueError("Invalid timezone-aware schedule_window") from exc
        for key in ("lease_seconds", "max_attempts", "retry_backoff_seconds", "max_concurrency"):
            self._positive(cfg.get(key), key)
        if cfg["max_concurrency"] != 1:
            raise ValueError("M4 conductor requires max_concurrency=1")
        if int(cfg["max_attempts"]) != cfg["max_attempts"]:
            raise ValueError("max_attempts must be an integer")
        for key in ("global_daily_budget", "bot_daily_budget", "job_budget"):
            for resource in RESOURCES:
                self._positive(cfg.get(key, {}).get(resource), f"{key}.{resource}")
        if not isinstance(cfg.get("bots"), list) or not cfg["bots"]:
            raise ValueError("At least one configured bot is required")
        ids = set()
        for bot in cfg["bots"]:
            if not isinstance(bot, dict):
                raise ValueError("Bot must be an object")
            if not isinstance(bot.get("id"), str) or not bot["id"].strip():
                raise ValueError("Bot ID must be a nonempty string")
            if not bot.get("id") or bot["id"] in ids:
                raise ValueError("Bot IDs must be unique and nonempty")
            ids.add(bot["id"])
            if not isinstance(bot.get("tenant_id"), str) or not bot["tenant_id"].strip():
                raise ValueError("Every bot requires an exact tenant scope")
            if cfg["mode"] == "replay" and not bot["tenant_id"].startswith("synthetic-"):
                raise ValueError("Replay bots require synthetic tenants")
            if not isinstance(bot.get("enabled"), bool):
                raise ValueError("Bot enabled must be boolean")
            if bot.get("kind") not in {"segment", "client"}:
                raise ValueError("Bot kind must be segment or client")
            if bot.get("segment") not in {
                "healthcare",
                "pe-technical-diligence",
                "alumni-associations",
                "client-bd",
            }:
                raise ValueError("Unknown research segment")
            if bot.get("kind") == "client" and bot.get("segment") != "client-bd":
                raise ValueError("Client bots must use the client-bd segment")
            if "research_segment" in bot and (
                not isinstance(bot["research_segment"], str)
                or bot["research_segment"] not in RESEARCH_SEGMENTS
            ):
                raise ValueError("Unknown client research_segment")
            if "managed_leads" in bot:
                managed = bot["managed_leads"]
                required = {"base_url", "client_id_env", "token_env", "workspace_id"}
                if (
                    not isinstance(managed, dict)
                    or set(managed) != required
                    or not all(
                        isinstance(value, str) and value.strip() for value in managed.values()
                    )
                ):
                    raise ValueError("managed_leads requires exact named string configuration")
                if (
                    managed["base_url"] != "https://managed-leads.vercel.app"
                    or managed["workspace_id"] != bot["tenant_id"]
                ):
                    raise ValueError(
                        "Managed Leads base URL and workspace must match configured scope"
                    )
                if not all(
                    re.fullmatch(r"[A-Z][A-Z0-9_]{0,100}", managed[key])
                    for key in ("client_id_env", "token_env")
                ):
                    raise ValueError("Managed Leads credentials must be environment variable names")
            hosts = bot.get("source_allowlist", [])
            if not isinstance(hosts, list):
                raise ValueError("source_allowlist must be a list of exact hostnames")
            for host in hosts:
                if (
                    not isinstance(host, str)
                    or len(host) > 253
                    or "." not in host
                    or not all(
                        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                        for label in host.split(".")
                    )
                    or host.endswith((".localhost", ".local", ".internal"))
                ):
                    raise ValueError("source_allowlist requires lowercase public DNS hostnames")
                try:
                    ipaddress.ip_address(host)
                except ValueError:
                    pass
                else:
                    raise ValueError("source_allowlist cannot contain IP addresses")
            schedule = bot.get("schedule")
            if not isinstance(schedule, dict) or not isinstance(schedule.get("enabled"), bool):
                raise ValueError("Bot schedule.enabled must be boolean")
            self._positive(schedule.get("interval_seconds"), "schedule.interval_seconds")
            if schedule["interval_seconds"] < 1:
                raise ValueError("Schedule interval must be at least one second")
            if "around_the_clock" in schedule and not isinstance(
                schedule.get("around_the_clock"), bool
            ):
                raise ValueError("Bot schedule.around_the_clock must be boolean")

    @staticmethod
    def _positive(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _event(self, db, kind, job=None, **data):
        mode = job["mode"] if job else self.config["mode"]
        details = {**data, "mode": mode, "replay_only": mode == "replay"}
        if job:
            details.update(bot_id=job["bot_id"], tenant_id=job["tenant_id"])
        for key in ("id", "timestamp", "kind", "job_id"):
            details.pop(key, None)
        db.execute(
            "INSERT INTO bd_events(timestamp,kind,job_id,data) VALUES(?,?,?,?)",
            (self.clock(), kind, job["id"] if job else None, json.dumps(details, allow_nan=False)),
        )

    @staticmethod
    def _save(db, job):
        db.execute(
            "UPDATE bd_jobs SET state=?, data=? WHERE id=?",
            (job["state"], json.dumps(job, allow_nan=False), job["id"]),
        )

    @staticmethod
    def _get(db, job_id):
        row = db.execute("SELECT data FROM bd_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return json.loads(row[0])

    @staticmethod
    def _killed(db):
        row = db.execute("SELECT value FROM bd_controls WHERE key='kill_switch'").fetchone()
        return bool(row and row[0])

    def enqueue(self, bot_id, tenant_id, entity_id, task, payload=None, mode=None):
        bot = self.bots.get(bot_id)
        if not bot or not bot["enabled"] or bot["tenant_id"] != tenant_id:
            raise ValueError("Bot must be enabled and scoped to the exact tenant")
        mode = mode or self.config["mode"]
        if mode != self.config["mode"]:
            raise ValueError("Job mode must match explicitly configured runtime mode")
        if not all(isinstance(v, str) and v.strip() for v in (entity_id, task)):
            raise ValueError("entity_id and task are required")
        now = self.clock()
        with self._transaction() as db:
            existing = db.execute(
                "SELECT data FROM bd_jobs WHERE tenant_id=? AND entity_id=? AND task=?",
                (tenant_id, entity_id, task),
            ).fetchone()
            if existing:
                return json.loads(existing[0])
            job = dict(
                id=uuid.uuid4().hex,
                bot_id=bot_id,
                tenant_id=tenant_id,
                entity_id=entity_id,
                task=task,
                payload=payload or {},
                state="queued",
                mode=mode,
                replay_only=mode == "replay",
                attempts=0,
                created_at=now,
                available_at=now,
                updated_at=now,
                lease_token=None,
                lease_until=None,
                heartbeat_at=None,
                step="queued",
                priority=0,
            )
            db.execute(
                "INSERT INTO bd_jobs VALUES(?,?,?,?,?,?)",
                (job["id"], tenant_id, entity_id, task, "queued", json.dumps(job, allow_nan=False)),
            )
            self._event(db, "job_queued", job)
            return job

    def _retry(self, db, job, error):
        now = self.clock()
        job.update(error=str(error)[:1000], updated_at=now, lease_token=None, lease_until=None)
        if job["attempts"] < self.config["max_attempts"]:
            job.update(
                state="queued",
                step="retry_wait",
                available_at=now
                + self.config["retry_backoff_seconds"] * 2 ** (job["attempts"] - 1),
            )
            kind = "job_retried"
        else:
            job.update(state="failed", step="failed", finished_at=now)
            kind = "job_failed"
        self._save(db, job)
        self._event(db, kind, job, error=job["error"])

    def _recover(self, db):
        for row in db.execute(
            "SELECT data FROM bd_jobs WHERE state IN ('leased','running')"
        ).fetchall():
            job = json.loads(row[0])
            if job["lease_until"] <= self.clock() or job["deadline_at"] <= self.clock():
                self._retry(db, job, "Lease expired or attempt wall-time exhausted")

    def recover(self):
        with self._transaction() as db:
            self._recover(db)

    def _reserve(self, db, bot_id):
        day = datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat()
        caps = self.config["job_budget"]
        scopes = (
            ("global", self.effective_budget("global_daily_budget", db=db)),
            ("bot:" + bot_id, self.effective_budget("bot_daily_budget", bot_id=bot_id, db=db)),
        )
        for scope, budget in scopes:
            for resource in RESOURCES:
                row = db.execute(
                    "SELECT reserved FROM bd_budget WHERE day=? AND scope=? AND resource=?",
                    (day, scope, resource),
                ).fetchone()
                if (row[0] if row else 0) + caps[resource] > budget[resource]:
                    return False
        for scope, _ in scopes:
            for resource in RESOURCES:
                db.execute(
                    "INSERT INTO bd_budget VALUES(?,?,?,?) ON CONFLICT(day,scope,resource) DO UPDATE SET reserved=reserved+excluded.reserved",
                    (day, scope, resource, caps[resource]),
                )
        return True

    def lease(self, worker_id):
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id is required")
        with self._transaction() as db:
            self._recover(db)
            if self._killed(db):
                return None
            active = db.execute(
                "SELECT COUNT(*) FROM bd_jobs WHERE state IN ('leased','running')"
            ).fetchone()[0]
            if active >= self.config.get("max_concurrency", 1):
                return None
            rows = db.execute(
                "SELECT rowid, data FROM bd_jobs WHERE state='queued' ORDER BY rowid"
            ).fetchall()
            ranked = []
            for rowid, payload in rows:
                job = json.loads(payload)
                ranked.append(
                    (-(job.get("priority") or 0), job.get("available_at") or 0, rowid, job)
                )
            ranked.sort()
            for _, _, _, job in ranked:
                bot = self.bots.get(job["bot_id"])
                if (
                    not bot
                    or not bot["enabled"]
                    or bot["tenant_id"] != job["tenant_id"]
                    or job["mode"] != self.config["mode"]
                    or job["available_at"] > self.clock()
                ):
                    continue
                if not self._reserve(db, job["bot_id"]):
                    day = datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat()
                    if job.get("budget_blocked_day") != day:
                        job.update(
                            step="daily_budget_exhausted",
                            budget_blocked_day=day,
                            updated_at=self.clock(),
                        )
                        self._save(db, job)
                        self._event(
                            db,
                            "daily_budget_exhausted",
                            job,
                            day=day,
                            message="Daily reservation budget exhausted; queued until budget resets",
                        )
                    continue
                now = self.clock()
                deadline = now + self.config["job_budget"]["seconds"]
                job.update(
                    state="leased",
                    step="leased",
                    attempts=job["attempts"] + 1,
                    lease_token=uuid.uuid4().hex,
                    worker_id=worker_id,
                    lease_until=min(now + self.config["lease_seconds"], deadline),
                    deadline_at=deadline,
                    heartbeat_at=now,
                    updated_at=now,
                    reserved_budget=self.config["job_budget"],
                    used_budget={"tokens": 0, "calls": 0},
                )
                self._save(db, job)
                self._event(db, "job_leased", job)
                return job
            return None

    def _fenced(self, db, job_id, lease_token):
        job = self._get(db, job_id)
        if (
            self._killed(db)
            or job["state"] not in ACTIVE
            or not lease_token
            or job["lease_token"] != lease_token
            or job["lease_until"] <= self.clock()
            or job["deadline_at"] <= self.clock()
        ):
            raise LeaseLost("Lease expired, cancelled, stopped, or owned by another worker")
        return job

    def start(self, job_id, lease_token):
        with self._transaction() as db:
            job = self._fenced(db, job_id, lease_token)
            if job["state"] != "leased":
                raise LeaseLost("Only leased jobs can start")
            job.update(
                state="running", step="running", started_at=self.clock(), updated_at=self.clock()
            )
            self._save(db, job)
            self._event(db, "job_started", job)
            return job

    def heartbeat(self, job_id, lease_token, step=None):
        with self._transaction() as db:
            job = self._fenced(db, job_id, lease_token)
            now = self.clock()
            job.update(
                heartbeat_at=now,
                updated_at=now,
                lease_until=min(now + self.config["lease_seconds"], job["deadline_at"]),
            )
            if step:
                job["step"] = str(step)[:100]
            self._save(db, job)
            return job

    def consume(self, job_id, lease_token, *, tokens=0, calls=0):
        """Check and debit an attempt before an adapter call, never after spending."""
        with self._transaction() as db:
            job = self._fenced(db, job_id, lease_token)
            for resource, amount in (("tokens", tokens), ("calls", calls)):
                if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                    raise ValueError("Consumption must be nonnegative integers")
                if job["used_budget"][resource] + amount > job["reserved_budget"][resource]:
                    raise ValueError("Attempt budget exhausted")
                job["used_budget"][resource] += amount
            self._save(db, job)
            return job

    def record_event(self, job_id, lease_token, kind, **data):
        with self._transaction() as db:
            job = self._fenced(db, job_id, lease_token)
            self._event(db, kind, job, **data)

    def complete(self, job_id, lease_token, result=None):
        with self._transaction() as db:
            job = self._fenced(db, job_id, lease_token)
            if job["state"] != "running":
                raise LeaseLost("Only running jobs can complete")
            job.update(
                state="completed",
                step="completed",
                result=result or {},
                finished_at=self.clock(),
                updated_at=self.clock(),
                lease_token=None,
                lease_until=None,
            )
            self._save(db, job)
            self._event(db, "job_completed", job)
            return job

    def fail(self, job_id, lease_token, error):
        with self._transaction() as db:
            job = self._fenced(db, job_id, lease_token)
            self._retry(db, job, error)
            return job

    def cancel(self, job_id):
        with self._transaction() as db:
            job = self._get(db, job_id)
            if job["state"] not in TERMINAL:
                job.update(
                    state="cancelled",
                    step="cancelled",
                    lease_token=None,
                    lease_until=None,
                    updated_at=self.clock(),
                    finished_at=self.clock(),
                )
                self._save(db, job)
                self._event(db, "job_cancelled", job)
            return job

    def set_kill_switch(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        with self._transaction() as db:
            db.execute(
                "INSERT INTO bd_controls VALUES('kill_switch',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (int(enabled),),
            )
            if enabled:
                for row in db.execute(
                    "SELECT data FROM bd_jobs WHERE state IN ('leased','running')"
                ).fetchall():
                    job = json.loads(row[0])
                    job.update(
                        state="cancelled",
                        step="stopped",
                        lease_token=None,
                        lease_until=None,
                        updated_at=self.clock(),
                        finished_at=self.clock(),
                    )
                    self._save(db, job)
                    self._event(db, "job_cancelled", job, reason="kill_switch")
            self._event(db, "kill_switch_changed", enabled=enabled)
            self._audit(db, "kill_switch", enabled=enabled)

    def schedule_enabled(self):
        with self._transaction() as db:
            return self._schedule_enabled(db)

    def _schedule_enabled(self, db):
        row = db.execute("SELECT value FROM bd_controls WHERE key='schedule_enabled'").fetchone()
        if row is not None:
            return bool(row[0])
        return bool(self.config["schedule_enabled"])

    def set_schedule_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        with self._transaction() as db:
            db.execute(
                "INSERT INTO bd_controls VALUES('schedule_enabled',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (int(enabled),),
            )
            self._event(db, "schedule_enabled_changed", enabled=enabled)
            self._audit(db, "resume" if enabled else "pause", enabled=enabled)

    def _steering_value(self, db, key, default=None):
        row = db.execute("SELECT value FROM bd_steering WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set_steering(self, db, key, value):
        db.execute(
            "INSERT INTO bd_steering VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, allow_nan=False)),
        )

    def _audit(self, db, action, **data):
        payload = {key: value for key, value in data.items() if value is not None}
        db.execute(
            "INSERT INTO bd_audit(timestamp,action,data) VALUES(?,?,?)",
            (self.clock(), action, json.dumps(payload, allow_nan=False)),
        )
        event_data = dict(payload)
        if "kind" in event_data:
            event_data["focus_kind"] = event_data.pop("kind")
        if "id" in event_data:
            event_data.setdefault("focus_id", event_data.pop("id"))
        job = None
        job_id = event_data.pop("job_id", None)
        if job_id:
            try:
                job = self._get(db, job_id)
            except KeyError:
                event_data["queued_job_id"] = job_id
        self._event(db, "steer", job=job, action=action, **event_data)

    def _require_bot(self, bot_id):
        bot = self.bots.get(bot_id)
        if not bot:
            raise KeyError("enabled_bot_not_found")
        return bot

    def steering_state(self, db=None):
        def read(conn):
            paused = self._steering_value(conn, "paused_bots", [])
            overlay_clock = self._steering_value(conn, "around_the_clock", {}) or {}
            clock = {
                bot_id: bool(overlay_clock[bot_id])
                if bot_id in overlay_clock
                else around_the_clock(bot)
                for bot_id, bot in self.bots.items()
            }
            cadence = self.effective_cadence(db=conn) or {}
            overnight = (
                cadence.get("overnight") if isinstance(cadence.get("overnight"), dict) else {}
            )
            return {
                "paused_bots": [bot_id for bot_id in paused if bot_id in self.bots],
                "focus": self._steering_value(conn, "focus"),
                "overnight_enabled": bool(overnight.get("enabled")),
                "around_the_clock": clock,
                "cadence_overlay": self._steering_value(conn, "cadence_overlay"),
                "budget_overlay": self._steering_value(conn, "budget_overlay"),
            }

        if db is not None:
            return read(db)
        with self._transaction() as conn:
            return read(conn)

    def effective_cadence(self, db=None):
        raw = self.config.get("cadence")
        if not raw:
            return None
        cadence = json.loads(json.dumps(raw))

        def merge(conn):
            overlay = self._steering_value(conn, "cadence_overlay") or {}
            overnight_flag = self._steering_value(conn, "overnight_enabled")
            if "weekday_interval_seconds" in overlay:
                cadence["weekday"]["interval_seconds"] = overlay["weekday_interval_seconds"]
            if "weekend_interval_seconds" in overlay:
                cadence["weekend"]["interval_seconds"] = overlay["weekend_interval_seconds"]
            if overnight_flag is not None or "overnight_interval_seconds" in overlay:
                cadence.setdefault("overnight", {"enabled": False, "interval_seconds": 3600})
            if overnight_flag is not None:
                cadence["overnight"]["enabled"] = bool(overnight_flag)
            if "overnight_interval_seconds" in overlay:
                cadence["overnight"]["interval_seconds"] = overlay["overnight_interval_seconds"]
            return cadence

        if db is not None:
            return merge(db)
        with self._transaction() as conn:
            return merge(conn)

    def effective_budget(self, key, bot_id=None, db=None):
        base = dict(self.config[key])

        def merge(conn):
            overlay = self._steering_value(conn, "budget_overlay") or {}
            patch = (
                overlay.get("global")
                if key == "global_daily_budget"
                else ((overlay.get("bots") or {}).get(bot_id) if bot_id else None)
            )
            if not patch:
                return base
            merged = dict(base)
            for resource in RESOURCES:
                if resource in patch:
                    merged[resource] = patch[resource]
            return merged

        if db is not None:
            return merge(db)
        with self._transaction() as conn:
            return merge(conn)

    def bot_paused(self, bot_id, db=None):
        state = self.steering_state(db)
        return bot_id in state["paused_bots"]

    def pause_bot(self, bot_id, paused=True):
        self._require_bot(bot_id)
        with self._transaction() as db:
            current = [
                item for item in self._steering_value(db, "paused_bots", []) if item in self.bots
            ]
            if paused and bot_id not in current:
                current.append(bot_id)
            if not paused:
                current = [item for item in current if item != bot_id]
            self._set_steering(db, "paused_bots", current)
            self._audit(db, "pause_bot" if paused else "resume_bot", bot_id=bot_id)

    def set_focus(self, *, bot_id=None, segment=None):
        if (bot_id is None) == (segment is None):
            raise ValueError("focus requires exactly one of bot_id or segment")
        if bot_id is not None:
            self._require_bot(bot_id)
            focus = {"kind": "bot", "id": bot_id}
        else:
            known = RESEARCH_SEGMENTS | {"client-bd"}
            if segment not in known:
                raise ValueError("unknown_focus_segment")
            focus = {"kind": "segment", "id": segment}
        with self._transaction() as db:
            self._set_steering(db, "focus", focus)
            self._audit(db, "focus", bot_id=bot_id, segment=segment, **focus)

    def clear_focus(self):
        with self._transaction() as db:
            self._set_steering(db, "focus", None)
            self._audit(db, "clear_focus")

    def bump_job(self, job_id):
        with self._transaction() as db:
            job = self._get(db, job_id)
            if job["state"] != "queued":
                raise ValueError("only_queued_jobs_can_be_bumped")
            highest = 0
            for row in db.execute("SELECT data FROM bd_jobs WHERE state='queued'").fetchall():
                highest = max(highest, json.loads(row[0]).get("priority") or 0)
            job["priority"] = highest + 1
            job["updated_at"] = self.clock()
            self._save(db, job)
            self._audit(db, "bump", job_id=job_id, bot_id=job["bot_id"], priority=job["priority"])
            return job

    def set_overnight(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        with self._transaction() as db:
            self._set_steering(db, "overnight_enabled", enabled)
            self._audit(db, "set_cadence", overnight_enabled=enabled)

    def set_around_the_clock(self, bot_id, enabled):
        self._require_bot(bot_id)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        with self._transaction() as db:
            overlay = dict(self._steering_value(db, "around_the_clock", {}) or {})
            overlay[bot_id] = enabled
            self._set_steering(db, "around_the_clock", overlay)
            self._audit(db, "set_cadence", bot_id=bot_id, around_the_clock=enabled)

    def set_cadence_intervals(self, **changes):
        allowed = {
            "weekday_interval_seconds",
            "weekend_interval_seconds",
            "overnight_interval_seconds",
        }
        extra = set(changes) - allowed
        if extra or not changes:
            raise ValueError("cadence overlay requires known interval fields")
        overlay = {}
        for key, value in changes.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 60 or value > 86400:
                raise ValueError(f"{key} must be an integer from 60 to 86400")
            overlay[key] = value
        if not overlay:
            raise ValueError("cadence overlay requires known interval fields")
        trial = json.loads(json.dumps(self.config.get("cadence") or {}))
        if not trial:
            raise ValueError("cadence is not configured")
        trial.setdefault("overnight", {"enabled": False, "interval_seconds": 3600})
        if "weekday_interval_seconds" in overlay:
            trial["weekday"]["interval_seconds"] = overlay["weekday_interval_seconds"]
        if "weekend_interval_seconds" in overlay:
            trial["weekend"]["interval_seconds"] = overlay["weekend_interval_seconds"]
        if "overnight_interval_seconds" in overlay:
            trial["overnight"]["interval_seconds"] = overlay["overnight_interval_seconds"]
        validate_cadence(trial)
        with self._transaction() as db:
            current = dict(self._steering_value(db, "cadence_overlay") or {})
            current.update(overlay)
            self._set_steering(db, "cadence_overlay", current)
            self._audit(db, "set_cadence", **overlay)

    def set_budget_overlay(self, *, bot_id=None, tokens=None, calls=None, seconds=None):
        patch = {
            key: value
            for key, value in (("tokens", tokens), ("calls", calls), ("seconds", seconds))
            if value is not None
        }
        if not patch:
            raise ValueError("budget overlay requires tokens, calls or seconds")
        for key, value in patch.items():
            self._positive(value, key)
            if type(value) is not int:
                raise ValueError(f"{key} must be a positive integer")
        ceiling_key = "bot_daily_budget" if bot_id else "global_daily_budget"
        if bot_id is not None:
            self._require_bot(bot_id)
        ceiling = self.config[ceiling_key]
        for key, value in patch.items():
            if value > ceiling[key]:
                raise ValueError("budget_exceeds_configured_ceiling")
        with self._transaction() as db:
            overlay = dict(self._steering_value(db, "budget_overlay") or {})
            if bot_id is None:
                overlay["global"] = {**overlay.get("global", {}), **patch}
            else:
                bots = dict(overlay.get("bots") or {})
                bots[bot_id] = {**bots.get(bot_id, {}), **patch}
                overlay["bots"] = bots
            self._set_steering(db, "budget_overlay", overlay)
            self._audit(db, "set_budget", bot_id=bot_id, **patch)

    def audit_log(self, limit=50):
        with self._transaction() as db:
            rows = db.execute(
                "SELECT id,timestamp,action,data FROM bd_audit ORDER BY id DESC LIMIT ?",
                (min(max(int(limit), 1), 200),),
            ).fetchall()
            reserved = {"id", "timestamp", "action"}
            log = []
            for row in rows:
                details = json.loads(row[3])
                if "id" in details:
                    details.setdefault("focus_id", details.pop("id"))
                log.append(
                    {
                        **{key: value for key, value in details.items() if key not in reserved},
                        "id": row[0],
                        "timestamp": row[1],
                        "action": row[2],
                    }
                )
            return log

    def evidence_keys(self, bot_id, tenant_id):
        bot = self.bots.get(bot_id)
        if not bot or bot["tenant_id"] != tenant_id:
            raise ValueError("Evidence lookup requires exact configured bot and tenant")
        keys = set()
        with self._transaction() as db:
            rows = db.execute(
                "SELECT data FROM bd_jobs WHERE tenant_id=? AND state='completed'",
                (tenant_id,),
            ).fetchall()
            for row in rows:
                job = json.loads(row[0])
                if job["bot_id"] != bot_id:
                    continue
                result = job.get("result") or {}
                for key in result.get("reviewed_evidence", []):
                    if isinstance(key, str) and re.fullmatch(r"[a-f0-9]{64}", key):
                        keys.add(key)
                for packet in result.get("packets", []):
                    key = packet.get("evidence_key")
                    if isinstance(key, str) and key:
                        keys.add(key)
                        continue
                    source_url = packet.get("source_url")
                    source_hash = (packet.get("provenance") or {}).get("source_hash")
                    if (
                        isinstance(source_url, str)
                        and source_url
                        and isinstance(source_hash, str)
                        and source_hash
                    ):
                        keys.add(
                            hashlib.sha256((source_url + "\0" + source_hash).encode()).hexdigest()
                        )
        return sorted(keys)

    def set_packet_review(self, job_id, index, status):
        allowed = {"human_review_required", "research_next", "watch", "dismissed"}
        if status not in allowed:
            raise ValueError("Invalid proposal review status")
        if type(index) is not int or index < 0:
            raise ValueError("Packet index must be a nonnegative integer")
        with self._transaction() as db:
            job = self._get(db, job_id)
            result = job.get("result") or {}
            packets = result.get("packets", [result] if result.get("organization") else [])
            if (
                job["state"] != "completed"
                or index >= len(packets)
                or not isinstance(packets[index], dict)
                or not packets[index].get("organization")
            ):
                raise ValueError("Review requires a completed proposal packet")
            reviews = job.setdefault("packet_reviews", {})
            reviews[str(index)] = {"status": status, "updated_at": self.clock()}
            self._save(db, job)
            self._event(
                db,
                "packet_reviewed",
                job,
                packet_index=index,
                review_status=status,
                message="Local research triage updated; no outreach or CRM action",
            )
            return reviews[str(index)]

    def events(self, after_id=0, limit=200):
        with self._transaction() as db:
            rows = db.execute(
                "SELECT id,timestamp,kind,job_id,data FROM bd_events WHERE id>? ORDER BY id LIMIT ?",
                (after_id, min(max(int(limit), 1), 1000)),
            ).fetchall()
            reserved = {"id", "timestamp", "kind", "job_id"}
            events = []
            for row in rows:
                details = json.loads(row[4])
                events.append(
                    {
                        **{key: value for key, value in details.items() if key not in reserved},
                        "id": row[0],
                        "timestamp": row[1],
                        "kind": row[2],
                        "job_id": row[3],
                    }
                )
            return events

    def snapshot(self):
        with self._transaction() as db:
            self._recover(db)
            jobs = [
                json.loads(r[0]) for r in db.execute("SELECT data FROM bd_jobs ORDER BY rowid DESC")
            ]
            budgets = [
                dict(day=r[0], scope=r[1], resource=r[2], reserved=r[3])
                for r in db.execute("SELECT day,scope,resource,reserved FROM bd_budget")
            ]
            for job in jobs:
                job.pop("lease_token", None)
            bots = []
            for bot in self.bots.values():
                counts = {
                    state: sum(j["bot_id"] == bot["id"] and j["state"] == state for j in jobs)
                    for state in ("queued", "leased", "running", "completed", "failed", "cancelled")
                }
                bots.append({**bot, "counts": counts, "budget": self.config["bot_daily_budget"]})
            steering = self.steering_state(db)
            return dict(
                mode=self.config["mode"],
                replay_only=self.config["mode"] == "replay",
                schedule_enabled=self._schedule_enabled(db),
                cadence=self.effective_cadence(db),
                kill_switch=self._killed(db),
                bots=bots,
                jobs=jobs,
                budgets=budgets,
                budget_policy="Full caps reserved per attempt; no refunds",
                budget_day_timezone="UTC",
                timestamp=self.clock(),
                steering=steering,
            )
