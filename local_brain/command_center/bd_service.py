"""Background BD worker and public activity projection, isolated from legacy jobs."""

import asyncio
import hashlib
import logging
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .bd_cadence import (
    around_the_clock as bot_around_the_clock,
)
from .bd_cadence import (
    cadence_interval,
    cadence_open,
    cadence_window,
    eligible_bots,
    next_cadence_timestamp,
    next_query_for,
    rotate_bot,
    slot_index,
    status_banner,
)
from .bd_conductor import BDConductor, LeaseLost

logger = logging.getLogger(__name__)


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def _verdict_label(packet):
    answers = packet.get("judgments") if isinstance(packet.get("judgments"), dict) else {}
    disposition = answers.get("disposition") if isinstance(answers.get("disposition"), dict) else {}
    kind = str(disposition.get("type") or "").strip().lower()
    return {"choice": "Choice", "score": "Score", "noul": "Noul"}.get(kind)


def _evidence_score(packet):
    answers = packet.get("judgments") if isinstance(packet.get("judgments"), dict) else {}
    evidence = (
        answers.get("evidence_strength")
        if isinstance(answers.get("evidence_strength"), dict)
        else {}
    )
    score = evidence.get("score")
    if type(score) in (int, float) and 0 <= score <= 2:
        return score
    return None


def candidate_assessment(packet, segment):
    """Classify retained evidence without promoting historical or human-triaged packets."""
    from .bd_pipeline import (
        SEGMENT_ENTITIES,
        entity_gap,
        questions_for,
        validate_answers,
        validate_extraction,
    )

    answers = packet.get("judgments") or {}
    if not isinstance(answers, dict):
        return "inconclusive", "source_evidence_unverified"
    disposition = answers.get("disposition") or {}
    role = answers.get("entity_role") or {}
    if packet.get("disposition") == "reject":
        return "rejected", "segment_rejected"
    if not str(packet.get("organization") or "").strip():
        return "inconclusive", "no_named_organization"
    if segment not in SEGMENT_ENTITIES:
        return "inconclusive", "segment_evidence_unverified"
    if packet.get("disposition") != "research":
        return "inconclusive", "segment_entity_or_event_unconfirmed"
    source = packet.get("provenance") or {}
    signals = packet.get("market_signals") or []
    if not source or not signals:
        return "inconclusive", "source_evidence_missing"
    if not isinstance(signals, list) or not isinstance(signals[0], dict):
        return "inconclusive", "source_evidence_unverified"
    signal = signals[0]
    extraction = {
        "organization": packet["organization"],
        "signal": signal.get("text", ""),
        "quote": signal.get("quote", ""),
        "consumer_insight": "",
    }
    try:
        validate_extraction(source, extraction)
        validate_answers(answers, questions_for(segment))
        gap = entity_gap(source, extraction, segment)
    except (ValueError, KeyError, TypeError, AttributeError):
        return "inconclusive", "source_evidence_unverified"
    if gap:
        return "inconclusive", gap
    expected = SEGMENT_ENTITIES[segment][0]
    if (
        role.get("choice") != expected
        or role.get("confidence", 0) < 0.8
        or (role.get("probabilities") or {}).get(expected, 0) < 0.8
        or disposition.get("choice") != "research"
        or (answers.get("claim_supported") or {}).get("noul", 0) < 0.8
    ):
        return "inconclusive", "segment_entity_or_event_unconfirmed"
    return "supported", "source_supported_candidate"


class BDService:
    def __init__(self, store: BDConductor, pipeline=None):
        self.store = store
        self.pipeline = pipeline
        self.worker_id = "bd-" + uuid.uuid4().hex
        self._active = None
        self._worker = None
        self.last_error = None
        self._qwen_ready_at = 0.0

    async def start(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self.loop())

    async def stop(self):
        if self._worker:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    def _standing(self):
        return bool(self.store.config.get("cadence")) and self.store.schedule_enabled()

    def _cadence(self):
        return self.store.effective_cadence() or self.store.config.get("cadence")

    def _steering(self):
        return self.store.steering_state()

    def _scheduled_bots(self, *, shadow=False, overnight=False, focus=None, paused=()):
        clock = self._steering().get("around_the_clock") or {}
        bots = []
        for bot in self.store.bots.values():
            item = dict(bot)
            schedule = dict(item.get("schedule") or {})
            if item["id"] in clock:
                schedule["around_the_clock"] = clock[item["id"]]
            item["schedule"] = schedule
            bots.append(item)
        return eligible_bots(
            bots,
            shadow=shadow,
            standing=self._standing() and not shadow,
            paused=paused,
            overnight=overnight,
            focus=None if shadow else focus,
        )

    def enqueue_cycle(self, *, shadow=False):
        mode = self.store.config["mode"]
        if shadow and mode != "replay":
            raise ValueError("Shadow runs require a separate replay database and configuration")
        if self.store.snapshot()["kill_switch"]:
            raise ValueError("Conductor is stopped; resume before queuing work")
        now = self.store.clock()
        cadence = self._cadence()
        steering = self._steering()
        window = cadence_window(cadence, now) if cadence and not shadow else None
        bots = self._scheduled_bots(
            shadow=shadow,
            overnight=window == "overnight",
            focus=steering.get("focus"),
            paused=steering.get("paused_bots") or (),
        )
        if not shadow and cadence:
            if not self.store.schedule_enabled() or not cadence_open(cadence, now):
                return []
            slot = slot_index(cadence, now)
            chosen = rotate_bot(bots, slot)
            bots = [chosen] if chosen else []
        queued = []
        for bot in bots:
            if not shadow and not cadence and not bot.get("schedule", {}).get("enabled", False):
                continue
            interval = (
                cadence_interval(cadence, now)
                if cadence and not shadow
                else bot.get("schedule", {}).get("interval_seconds", 3600)
            )
            slot = int(now // interval)
            # A deterministic window makes repeated clicks and scheduler restarts idempotent.
            entity = f"{bot['id']}:{slot}"
            queued.append(
                self.store.enqueue(
                    bot["id"],
                    bot["tenant_id"],
                    entity,
                    "public-research",
                    payload={
                        "segment": bot["segment"],
                        "source_job": bot.get("source_job", "lead_signals"),
                        "research_segment": bot.get("research_segment"),
                    },
                    mode=mode,
                )["id"]
            )
        return queued

    def queue_target(self, bot_id, *, query=None, target=None):
        from .bd_pipeline import DataUnavailable, operator_search_text

        bot = self.store.bots.get(bot_id)
        if not bot or not bot["enabled"]:
            raise KeyError("enabled_bot_not_found")
        if self.store.snapshot()["kill_switch"]:
            raise ValueError("Conductor is stopped; resume before queuing work")
        try:
            text = operator_search_text(query, target)
        except DataUnavailable as exc:
            raise ValueError(exc.code) from exc
        if not text:
            raise ValueError("queue requires a query or target")
        entity = "operator-queue:" + hashlib.sha256(text.encode()).hexdigest()[:24]
        job = self.store.enqueue(
            bot["id"],
            bot["tenant_id"],
            entity,
            "public-research",
            payload={
                "segment": bot["segment"],
                "source_job": bot.get("source_job", "lead_signals"),
                "research_segment": bot.get("research_segment"),
                "operator_query": query,
                "operator_target": target,
            },
        )
        if job["state"] == "queued":
            self.store.bump_job(job["id"])
        with self.store._transaction() as db:
            self.store._audit(
                db, "queue", bot_id=bot_id, query=query, target=target, job_id=job["id"]
            )
        return job

    def steer(self, action, **fields):
        if action == "pause":
            self.set_schedule_enabled(False)
        elif action == "resume":
            self.set_schedule_enabled(True)
        elif action == "pause_bot":
            self.store.pause_bot(fields["bot_id"], True)
        elif action == "resume_bot":
            self.store.pause_bot(fields["bot_id"], False)
        elif action == "focus":
            self.store.set_focus(bot_id=fields.get("bot_id"), segment=fields.get("segment"))
        elif action == "clear_focus":
            self.store.clear_focus()
        elif action == "bump":
            self.store.bump_job(fields["job_id"])
        elif action == "queue":
            self.queue_target(
                fields["bot_id"], query=fields.get("query"), target=fields.get("target")
            )
        elif action == "set_cadence":
            if fields.get("overnight_enabled") is not None:
                self.store.set_overnight(fields["overnight_enabled"])
            if fields.get("around_the_clock") is not None:
                self.store.set_around_the_clock(fields["bot_id"], fields["around_the_clock"])
            intervals = {
                key: fields[key]
                for key in (
                    "weekday_interval_seconds",
                    "weekend_interval_seconds",
                    "overnight_interval_seconds",
                )
                if fields.get(key) is not None
            }
            if intervals:
                self.store.set_cadence_intervals(**intervals)
            if (
                fields.get("overnight_enabled") is None
                and fields.get("around_the_clock") is None
                and not intervals
            ):
                raise ValueError("cadence overlay requires known interval fields")
        elif action == "set_budget":
            self.store.set_budget_overlay(
                bot_id=fields.get("bot_id"),
                tokens=fields.get("tokens"),
                calls=fields.get("calls"),
                seconds=fields.get("seconds"),
            )
        elif action == "kill_switch":
            self.set_kill_switch(fields["enabled"])
        elif action == "cancel":
            job = self.store.cancel(fields["job_id"])
            if job["state"] != "cancelled":
                raise ValueError("job_not_cancellable")
            with self.store._transaction() as db:
                self.store._audit(db, "cancel", job_id=job["id"], bot_id=job.get("bot_id"))
        else:
            raise ValueError("unknown_steer_action")
        return self.snapshot()

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
        job["research_segment"] = self.store.bots.get(job["bot_id"], {}).get("research_segment")
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

    def set_schedule_enabled(self, enabled):
        self.store.set_schedule_enabled(enabled)

    async def _ensure_qwen_ready(self):
        now = self.store.clock()
        if now - self._qwen_ready_at < 60:
            return
        from .bd_pipeline import ensure_qwen_loaded

        await ensure_qwen_loaded()
        self._qwen_ready_at = now

    async def loop(self):
        while True:
            try:
                raw = self.store.snapshot()
                if raw["schedule_enabled"] and not raw["kill_switch"]:
                    if raw["mode"] == "live":
                        try:
                            await self._ensure_qwen_ready()
                        except Exception as exc:
                            self.last_error = type(exc).__name__
                            logger.warning("BD qwen warmup unavailable: %s", self.last_error)
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
        """Standing cadence drives execution; a one-off window is display-only."""
        cadence = self._cadence()
        window = None if cadence else self.store.config.get("schedule_window")
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
        steering = raw.get("steering") or self._steering()
        bots = self._scheduled_bots(
            focus=steering.get("focus"),
            paused=steering.get("paused_bots") or (),
        )
        projection = {
            "state": "scheduled",
            "next_run_at": None,
            "start_at": iso(start),
            "stop_at": iso(end),
            "timezone": (
                cadence["timezone"]
                if cadence
                else window["timezone"]
                if window
                else "America/Chicago"
            ),
            "display_only": not bool(cadence),
        }
        if cadence:
            if not raw["schedule_enabled"] or not bots:
                projection["state"] = "disabled"
                return projection
            if raw["kill_switch"]:
                projection["state"] = "paused"
                return projection
            existing = {(job["tenant_id"], job["entity_id"], job["task"]) for job in raw["jobs"]}
            overnight_bots = [
                bot
                for bot in bots
                if (steering.get("around_the_clock") or {}).get(
                    bot["id"], bot_around_the_clock(bot)
                )
            ]
            nxt = next_cadence_timestamp(
                cadence, now, existing, bots, overnight_eligible=overnight_bots
            )
            if nxt is not None:
                projection["next_run_at"] = iso(nxt)
                if not cadence_open(cadence, now):
                    projection["state"] = "waiting"
                    projection["start_at"] = iso(nxt)
            return projection
        if end is not None and now >= end:
            projection["state"] = "expired"
            return projection
        if not raw["schedule_enabled"] or not bots:
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

    def _daily_summary(self, raw, jobs, packets):
        zone_name = (raw.get("cadence") or self.store.config.get("cadence") or {}).get(
            "timezone", "America/Chicago"
        )
        try:
            zone = ZoneInfo(zone_name)
        except ZoneInfoNotFoundError:
            zone = timezone.utc
            zone_name = "UTC"

        def local_day(value):
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return None
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(zone).date().isoformat()
            return datetime.fromtimestamp(value, timezone.utc).astimezone(zone).date().isoformat()

        today = (
            datetime.fromtimestamp(raw["timestamp"], timezone.utc)
            .astimezone(zone)
            .date()
            .isoformat()
        )
        today_jobs = [
            job
            for job in raw["jobs"]
            if local_day(job.get("finished_at") or job.get("created_at")) == today
        ]
        public_today = {
            job["id"]: job
            for job in jobs
            if local_day(job.get("finished_at") or job.get("created_at")) == today
        }
        proposed = sum(
            1 for job in public_today.values() if job.get("research_outcome") == "supported"
        )
        inconclusive = sum(
            1 for job in public_today.values() if job.get("research_outcome") == "inconclusive"
        )
        failed = sum(1 for job in today_jobs if job["state"] == "failed")
        awaiting = sum(
            1
            for packet in packets
            if packet.get("candidate_status") == "supported"
            and packet.get("review_status") == "human_review_required"
            and local_day(packet.get("created_at")) == today
        )
        return {
            "day": today,
            "timezone": zone_name,
            "candidates_proposed": proposed,
            "inconclusive": inconclusive,
            "failed": failed,
            "awaiting_human_review": awaiting,
        }

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

        steering = raw.get("steering") or self._steering()
        bots = []
        for bot in raw["bots"]:
            jobs = [j for j in raw["jobs"] if j["bot_id"] == bot["id"]]
            bots.append(
                {
                    **{
                        key: bot.get(key)
                        for key in (
                            "id",
                            "name",
                            "segment",
                            "tenant_id",
                            "enabled",
                            "kind",
                            "research_segment",
                        )
                    },
                    "paused": bot["id"] in (steering.get("paused_bots") or []),
                    "around_the_clock": bool(
                        (steering.get("around_the_clock") or {}).get(bot["id"])
                    ),
                    "focused": bool(
                        steering.get("focus")
                        and (
                            (
                                steering["focus"].get("kind") == "bot"
                                and steering["focus"].get("id") == bot["id"]
                            )
                            or (
                                steering["focus"].get("kind") == "segment"
                                and steering["focus"].get("id")
                                in {bot.get("segment"), bot.get("research_segment")}
                            )
                        )
                    ),
                    "counts": {
                        "queued": sum(j["state"] == "queued" for j in jobs),
                        "running": sum(j["state"] in ("leased", "running") for j in jobs),
                        "completed": sum(j["state"] == "completed" for j in jobs),
                        "failed": sum(j["state"] == "failed" for j in jobs),
                    },
                    "budgets": budget(
                        "bot:" + bot["id"],
                        self.store.effective_budget("bot_daily_budget", bot_id=bot["id"]),
                    ),
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
                for key in (
                    "id",
                    "bot_id",
                    "tenant_id",
                    "state",
                    "step",
                    "attempts",
                    "error",
                    "priority",
                )
            }
            payload = job.get("payload") or {}
            public["operator_query"] = payload.get("operator_query")
            public["operator_target"] = payload.get("operator_target")
            public["stage"] = self._job_stage(job)
            public["segment"] = self.store.bots.get(job["bot_id"], {}).get("segment", "unknown")
            public.update(
                {
                    key: iso(job.get(key))
                    for key in ("created_at", "started_at", "heartbeat_at", "finished_at")
                }
            )
            if job["state"] == "completed":
                result = job.get("result") or {}
                retained = result.get("packets", [result] if result.get("organization") else [])
                assessments = [
                    candidate_assessment(
                        packet,
                        self.store.bots.get(job["bot_id"], {}).get("research_segment")
                        or public["segment"],
                    )
                    for packet in retained
                ]
                outcome = result.get("research_outcome", "inconclusive")
                reason = result.get("reason", "no_supported_prospect")
                if assessments:
                    outcome, reason = next(
                        (item for item in assessments if item[0] == "supported"), assessments[0]
                    )
                public["research_outcome"] = (
                    outcome
                    if outcome in {"supported", "rejected", "inconclusive"}
                    else "inconclusive"
                )
                public["result_reason"] = reason
                public["next_query"] = result.get("next_query") or (
                    next_query_for(reason)
                    if public["research_outcome"] in {"inconclusive", "rejected"}
                    else None
                )
            elif job["state"] == "failed":
                public["next_query"] = next_query_for(job.get("error") or "")
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
            "cadence": raw.get("cadence"),
            "schedule": self._schedule_projection(raw),
            "status_banner": status_banner(raw["kill_switch"], raw["schedule_enabled"]),
            "daily_summary": self._daily_summary(raw, jobs, packets),
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
            "budget": budget("global", self.store.effective_budget("global_daily_budget")),
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
            "steering": {
                **steering,
                "audit": [
                    {
                        "id": item["id"],
                        "action": item["action"],
                        "created_at": iso(item["timestamp"]),
                        **{
                            key: item.get(key)
                            for key in (
                                "bot_id",
                                "job_id",
                                "kind",
                                "focus_id",
                                "query",
                                "target",
                                "enabled",
                                "overnight_enabled",
                                "around_the_clock",
                                "priority",
                            )
                            if item.get(key) is not None
                        },
                    }
                    for item in self.store.audit_log()
                ],
            },
            "assignments": self._assignments(raw, jobs, packets, events, now),
        }

    def _packet(self, job, packet, index):
        def strings(key):
            return [
                str(value) for value in packet.get(key, []) if isinstance(value, (str, int, float))
            ]

        segment = self.store.bots.get(job["bot_id"], {}).get("segment", "unknown")
        status, reason = candidate_assessment(
            packet, self.store.bots.get(job["bot_id"], {}).get("research_segment") or segment
        )
        return {
            "candidate_status": status,
            "candidate_reason": reason,
            "next_query": next_query_for(reason) if status == "inconclusive" else None,
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
            "permit": packet.get("permit"),
            "verdict": _verdict_label(packet),
            "evidence_score": _evidence_score(packet),
            "extraction_model": packet.get("extraction_model"),
            "judgment_model": packet.get("judgment_model"),
        }

    def _job_stage(self, job):
        step = str(job.get("step") or "")
        if job["state"] in {"queued", "leased"}:
            return "queued"
        if job["state"] == "running":
            if "jev" in step:
                return "judge"
            return "extract"
        if job["state"] == "completed":
            return "awaiting_human"
        return job["state"]

    def _assignments(self, raw, jobs, packets, events, _now=None):
        replay = raw["mode"] == "replay"

        def provider_block(name, assigned, waiting):
            kinds = {
                f"{name}_called",
                f"{name}_completed",
                f"{name}_error",
                f"{name}_replayed",
            }
            recent = [event for event in events if event["kind"] in kinds]
            live = [event for event in recent if event.get("mode") == "live"]
            completed = [event for event in live if event["kind"] == f"{name}_completed"]
            measured = [
                event["latency_ms"]
                for event in completed
                if type(event.get("latency_ms")) in (int, float)
            ]
            errors = sum(event["kind"] == f"{name}_error" for event in live)
            observed = bool(live) or replay
            return {
                "assigned": assigned,
                "queue": waiting,
                "recent": [
                    {
                        "id": event["id"],
                        "kind": event["kind"],
                        "job_id": event.get("job_id"),
                        "bot_id": event.get("bot_id"),
                        "created_at": iso(event["timestamp"]),
                        "latency_ms": event.get("latency_ms")
                        if type(event.get("latency_ms")) in (int, float)
                        else None,
                    }
                    for event in reversed(recent[-8:])
                ],
                "status": (
                    "running"
                    if assigned
                    else "replay"
                    if replay
                    else "idle"
                    if completed
                    else "data_unavailable"
                ),
                "calls": len(live) if observed else None,
                "errors": errors if observed else None,
                "latency_ms": (sum(measured) / len(measured)) if measured else None,
            }

        extract = [job["id"] for job in jobs if job.get("stage") == "extract"]
        judge = [job["id"] for job in jobs if job.get("stage") == "judge"]
        queued = [job["id"] for job in jobs if job.get("stage") == "queued"]
        awaiting = [
            packet["id"]
            for packet in packets
            if packet.get("candidate_status") == "supported"
            and packet.get("review_status") == "human_review_required"
        ]
        verdicts = [event for event in events if event["kind"] == "jev_verdict"]
        mix = (
            {
                key: sum(event.get("primitive", "").lower() == key.lower() for event in verdicts)
                for key in ("Choice", "Score", "Noul")
            }
            if verdicts or replay
            else None
        )
        jev = provider_block("jev", judge, [])
        jev["verdict_mix"] = mix
        jev["queue"] = [
            job["id"] for job in jobs if job.get("stage") == "extract" and job["state"] == "running"
        ]
        return {
            "qwen": provider_block("qwen", extract, queued),
            "jev": jev,
            "pipeline": [{"id": job_id, "stage": "extract"} for job_id in extract]
            + [{"id": job_id, "stage": "judge"} for job_id in judge]
            + [{"id": packet_id, "stage": "awaiting_human"} for packet_id in awaiting[:20]],
        }
