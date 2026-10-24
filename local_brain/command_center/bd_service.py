"""Background BD worker and public activity projection, isolated from legacy jobs."""

import asyncio
import logging
import uuid
from contextlib import suppress
from datetime import datetime, timezone

from .bd_conductor import BDConductor, LeaseLost

logger = logging.getLogger(__name__)


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


class BDService:
    def __init__(self, store: BDConductor, pipeline=None):
        self.store = store
        self.pipeline = pipeline
        self.worker_id = "bd-" + uuid.uuid4().hex
        self._active = None
        self._worker = None
        self.last_error = None

    async def start(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self.loop())

    async def stop(self):
        if self._worker:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    def enqueue_cycle(self, *, shadow=False):
        mode = self.store.config["mode"]
        if shadow and mode != "replay":
            raise ValueError("Shadow runs require a separate replay database and configuration")
        if self.store.snapshot()["kill_switch"]:
            raise ValueError("Conductor is stopped; resume before queuing work")
        queued = []
        now = self.store.clock()
        for bot in self.store.bots.values():
            if not bot["enabled"] or (
                not shadow and not bot.get("schedule", {}).get("enabled", False)
            ):
                continue
            interval = bot.get("schedule", {}).get("interval_seconds", 3600)
            # A deterministic window makes repeated clicks and scheduler restarts idempotent.
            entity = f"{bot['id']}:{int(now // interval)}"
            queued.append(
                self.store.enqueue(
                    bot["id"],
                    bot["tenant_id"],
                    entity,
                    "public-research",
                    payload={
                        "segment": bot["segment"],
                        "source_job": bot.get("source_job", "lead_signals"),
                    },
                    mode=mode,
                )["id"]
            )
        return queued

    async def run_once(self):
        job = self.store.lease(self.worker_id)
        if job is None:
            return False
        token = job["lease_token"]
        self.store.start(job["id"], token)
        job["segment"] = self.store.bots.get(job["bot_id"], {}).get("segment", "unknown")
        job["budget"] = job["reserved_budget"]
        job["source_allowlist"] = list(self.store.bots[job["bot_id"]].get("source_allowlist", []))
        job["managed_leads"] = dict(self.store.bots[job["bot_id"]].get("managed_leads", {}))
        job["research_segment"] = self.store.bots[job["bot_id"]].get("research_segment")
        job["seen_evidence"] = self.store.evidence_keys(job["bot_id"], job["tenant_id"])

        async def emit(kind, detail=None, **extra):
            data = {**(detail or {}), **extra}
            self.store.heartbeat(job["id"], token, step=data.get("step", kind))
            # Adapters must reserve a bounded call BEFORE making it.
            if data.get("reserve_tokens") or data.get("reserve_calls"):
                self.store.consume(
                    job["id"],
                    token,
                    tokens=data.pop("reserve_tokens", 0),
                    calls=data.pop("reserve_calls", 0),
                )
            self.store.record_event(job["id"], token, kind, **data)

        async def pulse():
            while True:
                await asyncio.sleep(min(5, self.store.config["lease_seconds"] / 3))
                self.store.heartbeat(job["id"], token)

        if self.pipeline is None:
            from .bd_pipeline import run_pipeline

            handler = run_pipeline
        else:
            handler = self.pipeline
        heartbeat = asyncio.create_task(pulse())
        work = asyncio.create_task(handler(job, emit=emit))
        self._active = work
        try:
            done, _ = await asyncio.wait(
                [work, heartbeat],
                timeout=job["reserved_budget"]["seconds"],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat in done:
                heartbeat.result()
                raise LeaseLost("heartbeat_stopped")
            if work not in done:
                raise TimeoutError("attempt_wall_time_exhausted")
            if work.cancelled():
                raise LeaseLost("job_stopped")
            result = work.result()
            self.store.complete(job["id"], token, result)
        except asyncio.CancelledError:
            # A service restart leaves a durable lease; another worker recovers it.
            raise
        except LeaseLost:
            pass
        except Exception as exc:
            # Provider exception strings can contain URLs, credentials or payloads.
            from .bd_pipeline import DataUnavailable

            reason = exc.code if isinstance(exc, DataUnavailable) else type(exc).__name__
            with suppress(LeaseLost):
                self.store.fail(job["id"], token, reason)
        finally:
            for task in (work, heartbeat):
                task.cancel()
            await asyncio.gather(work, heartbeat, return_exceptions=True)
            self._active = None
        return True

    def set_kill_switch(self, enabled):
        self.store.set_kill_switch(enabled)
        if enabled and self._active:
            self._active.cancel()

    async def loop(self):
        while True:
            try:
                if (
                    self.store.config["schedule_enabled"]
                    and not self.store.snapshot()["kill_switch"]
                ):
                    self.enqueue_cycle()
                await self.run_once()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = type(exc).__name__
                logger.warning("BD worker unavailable: %s", self.last_error)
            await asyncio.sleep(0.5)

    def _schedule_projection(self, raw):
        """Display external-guard metadata; this does not activate or stop scheduling."""
        window = self.store.config.get("schedule_window")
        start = (
            datetime.fromisoformat(window["start_at"].replace("Z", "+00:00")).timestamp()
            if window
            else None
        )
        end = (
            datetime.fromisoformat(window["end_at"].replace("Z", "+00:00")).timestamp()
            if window
            else None
        )
        now = raw["timestamp"]
        bots = [
            bot for bot in self.store.bots.values() if bot["enabled"] and bot["schedule"]["enabled"]
        ]
        projection = {
            "state": "scheduled",
            "next_run_at": None,
            "start_at": iso(start),
            "stop_at": iso(end),
            "timezone": window["timezone"] if window else "UTC",
            "display_only": True,
        }
        if end is not None and now >= end:
            projection["state"] = "expired"
            return projection
        if not self.store.config["schedule_enabled"] or not bots:
            projection["state"] = "disabled"
            return projection
        if start is not None and now < start:
            projection["state"] = "waiting"
        elif raw["kill_switch"]:
            projection["state"] = "paused"
            return projection
        existing = {(job["tenant_id"], job["entity_id"], job["task"]) for job in raw["jobs"]}
        candidates = []
        for bot in bots:
            interval = bot["schedule"]["interval_seconds"]
            candidate = max(now, start) if start is not None else now
            index = int(candidate // interval)
            while (bot["tenant_id"], f"{bot['id']}:{index}", "public-research") in existing:
                index += 1
                candidate = index * interval
            if end is None or candidate < end:
                candidates.append(candidate)
        if candidates:
            projection["next_run_at"] = iso(min(candidates))
        return projection

    def snapshot(self):
        raw = self.store.snapshot()
        now = raw["timestamp"]
        today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()

        def budget(scope, limits):
            rows = {
                r["resource"]: r["reserved"]
                for r in raw["budgets"]
                if r["day"] == today and r["scope"] == scope
            }
            return {
                key: {"used": rows.get(key, 0), "limit": limit} for key, limit in limits.items()
            }

        bots = []
        for bot in raw["bots"]:
            jobs = [j for j in raw["jobs"] if j["bot_id"] == bot["id"]]
            bots.append(
                {
                    **{
                        key: bot.get(key)
                        for key in ("id", "name", "segment", "tenant_id", "enabled", "kind")
                    },
                    "counts": {
                        "queued": sum(j["state"] == "queued" for j in jobs),
                        "running": sum(j["state"] in ("leased", "running") for j in jobs),
                        "completed": sum(j["state"] == "completed" for j in jobs),
                        "failed": sum(j["state"] == "failed" for j in jobs),
                    },
                    "budgets": budget("bot:" + bot["id"], self.store.config["bot_daily_budget"]),
                }
            )
        jobs = []
        packets = []
        active_jobs = [job for job in raw["jobs"] if job["state"] in ("leased", "running")]
        queued_jobs = sorted(
            [job for job in reversed(raw["jobs"]) if job["state"] == "queued"],
            key=lambda job: job["created_at"],
        )
        terminal_jobs = [
            job for job in raw["jobs"] if job["state"] in ("completed", "failed", "cancelled")
        ]
        selected_jobs = (active_jobs + queued_jobs + terminal_jobs)[:200]
        job_counts = {
            state: sum(job["state"] == state for job in raw["jobs"])
            for state in ("queued", "leased", "running", "completed", "failed", "cancelled")
        }
        for job in selected_jobs:
            public = {
                key: job.get(key)
                for key in ("id", "bot_id", "tenant_id", "state", "step", "attempts", "error")
            }
            public["segment"] = self.store.bots.get(job["bot_id"], {}).get("segment", "unknown")
            public.update(
                {
                    key: iso(job.get(key))
                    for key in ("created_at", "started_at", "heartbeat_at", "finished_at")
                }
            )
            jobs.append(public)
        for job in [job for job in raw["jobs"] if job["state"] == "completed"][:200]:
            result = job.get("result") or {}
            for index, packet in enumerate(
                result.get("packets", [result] if result.get("organization") else [])
            ):
                packets.append(self._packet(job, packet, index))
        # Bounded newest-first feed; event history remains durable in SQLite.
        with self.store._transaction() as db:
            last = db.execute("SELECT COALESCE(MAX(id),0) FROM bd_events").fetchone()[0]
        events = self.store.events(max(0, last - 200))
        feed = [
            {
                "id": e["id"],
                "kind": e["kind"],
                "message": str(
                    e.get("message")
                    or e.get("reason")
                    or e.get("error")
                    or e["kind"].replace("_", " ")
                ),
                "created_at": iso(e["timestamp"]),
                "bot_id": e.get("bot_id"),
                "job_id": e.get("job_id"),
                **{
                    key: e.get(key)
                    for key in ("provider", "model", "latency_ms", "mode", "primitive")
                },
            }
            for e in reversed(events)
        ]
        verdicts = [e for e in events if e["kind"] == "jev_verdict"]
        calls = [e for e in events if e["kind"] in ("jev_called", "jev_replayed")]
        latencies = [
            e["latency_ms"]
            for e in events
            if e["kind"] in ("jev_completed", "jev_replayed")
            and isinstance(e.get("latency_ms"), (int, float))
        ]
        replay = raw["mode"] == "replay"
        health = {
            "qwen": {
                "status": "replay" if replay else "data_unavailable",
                "detail": "Recorded extraction; no model call"
                if replay
                else "No successful provider observation yet",
            },
            "jev": {
                "status": "replay" if replay else "data_unavailable",
                "detail": "Recorded typed judgments; no API call"
                if replay
                else "No successful provider observation yet",
            },
            "managed_leads": {
                "status": "data_unavailable",
                "detail": "Tenant read adapter is not configured or verified; contact dates remain unavailable",
            },
        }
        for event in events:
            if event.get("provider") in ("qwen", "jev", "managed_leads") and event.get(
                "provider_status"
            ) in (
                "available",
                "data_unavailable",
            ):
                health[event["provider"]] = {
                    "status": "stale"
                    if event["provider_status"] == "available" and now - event["timestamp"] > 120
                    else event["provider_status"],
                    "detail": "Last successful provider observation is stale; no current probe"
                    if event["provider_status"] == "available" and now - event["timestamp"] > 120
                    else event.get("message", ""),
                    "observed_at": iso(event["timestamp"]),
                    "last_status": event["provider_status"],
                }
        model_activity = {}
        for provider in ("qwen", "jev"):
            provider_events = [
                event
                for event in events
                if event["kind"]
                in {
                    f"{provider}_called",
                    f"{provider}_completed",
                    f"{provider}_error",
                    f"{provider}_replayed",
                }
            ]
            actual = [event for event in provider_events if event.get("mode") == "live"]
            completed = [event for event in actual if event["kind"] == f"{provider}_completed"]
            active_jobs = [
                job["id"]
                for job in raw["jobs"]
                if job["mode"] == "live"
                and job["state"] in ("leased", "running")
                and job.get("step")
                in (
                    f"{provider}_called",
                    "local_qwen_extraction" if provider == "qwen" else "jev_called",
                )
                and job.get("heartbeat_at") is not None
                and now - job["heartbeat_at"] <= self.store.config["lease_seconds"]
            ]
            measured = [
                event["latency_ms"]
                for event in completed
                if type(event.get("latency_ms")) in (int, float)
            ]
            count = (
                sum(event["kind"] == f"{provider}_called" for event in actual)
                if actual
                else (0 if replay else None)
            )
            model_activity[provider] = {
                "mode": raw["mode"],
                "metrics_scope": "latest_200_events",
                "status": "running"
                if active_jobs
                else "replay"
                if replay
                else "idle"
                if health[provider]["status"] == "available"
                else "data_unavailable",
                "active_jobs": active_jobs,
                "calls": count,
                "attempts": count,
                "completed": len(completed) if actual or replay else None,
                "errors": sum(event["kind"] == f"{provider}_error" for event in actual)
                if actual or replay
                else None,
                "replayed": sum(
                    event["kind"] == f"{provider}_replayed" for event in provider_events
                ),
                "last_event_at": iso(provider_events[-1]["timestamp"]) if provider_events else None,
                "latency_ms": sum(measured) / len(measured) if measured else None,
            }
        return {
            "mode": raw["mode"],
            "schedule_enabled": raw["schedule_enabled"],
            "schedule": self._schedule_projection(raw),
            "kill_switch": raw["kill_switch"],
            "bots": bots,
            "jobs": jobs,
            "job_counts": job_counts,
            "job_window": {
                "limit": 200,
                "total": len(raw["jobs"]),
                "returned": len(jobs),
                "truncated": len(raw["jobs"]) > len(jobs),
                "queue_total": job_counts["queued"] + job_counts["leased"],
                "queue_returned": sum(
                    job["state"] in ("queued", "leased") for job in selected_jobs
                ),
                "selection": "active_then_oldest_queue_then_recent_terminal",
                "packet_history": "latest_200_completed_jobs",
            },
            "events": feed,
            "packets": packets,
            "health": health,
            "model_activity": model_activity,
            "budget": budget("global", self.store.config["global_daily_budget"]),
            "budget_policy": raw["budget_policy"],
            "budget_day_timezone": "UTC",
            "jev": {
                "calls_per_minute": sum(e["timestamp"] >= now - 60 for e in calls)
                if calls or replay
                else None,
                "verdict_mix": {
                    key: sum(e.get("primitive", "").lower() == key.lower() for e in verdicts)
                    for key in ("Choice", "Score", "Noul")
                }
                if verdicts or replay
                else None,
                "latency_ms": sum(latencies) / len(latencies) if latencies else None,
                "errors": sum(e["kind"] == "jev_error" for e in events)
                if calls or replay
                else None,
            },
            "worker_error": self.last_error,
            "metrics_scope": "latest_200_events",
        }

    def _packet(self, job, packet, index):
        def strings(key):
            return [
                str(value) for value in packet.get(key, []) if isinstance(value, (str, int, float))
            ]

        return {
            "id": packet.get("id", f"{job['id']}:{index}"),
            "job_id": job["id"],
            "bot_id": job["bot_id"],
            "tenant_id": job["tenant_id"],
            "segment": self.store.bots.get(job["bot_id"], {}).get("segment", "unknown"),
            "created_at": iso(job.get("finished_at", job["created_at"])),
            "mode": job["mode"],
            "index": index,
            "review_status": job.get("packet_reviews", {})
            .get(str(index), {})
            .get("status", "human_review_required"),
            "contact_policy": packet.get("contact_policy", "hold_pending_tenant_restriction_check"),
            "disposition": packet.get("disposition", "uncertain"),
            "source_quote": next(
                (
                    item.get("quote", "")
                    for item in packet.get("market_signals", [])
                    if isinstance(item, dict)
                ),
                "",
            ),
            "insight_status": "unverified_hypothesis",
            "organization": packet.get("organization", "Research packet"),
            "source_url": packet.get("source_url", ""),
            "emails": strings("emails"),
            "phones": strings("phones"),
            "signals": strings("signals"),
            "insights": strings("insights"),
            "summary": packet.get("summary", ""),
        }
