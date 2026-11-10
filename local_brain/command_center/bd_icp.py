"""Private per-bot ICP profiles and do-not-prospect lists, loaded from outside git.

Files live on the Mini under ~/.aiia-private/bd-icp/<bot_id>/ (mode 0700).
Missing files leave targeting unchanged. Malformed files fail closed for that
bot: they are unused, logged without contents, and surfaced on the snapshot.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_ICP_DIR = "AIIA_BD_ICP_DIR"
SUPPRESSED_REASON = "suppressed_existing_customer_pipeline"
_BOT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}")
_LEGAL_SUFFIXES = {
    "inc",
    "llc",
    "ltd",
    "co",
    "corp",
    "corporation",
    "incorporated",
    "limited",
    "company",
    "lp",
    "pllc",
    "pc",
    "plc",
}
_active_profile = contextvars.ContextVar("bd_icp_profile", default=None)


def default_icp_dir():
    return Path.home() / ".aiia-private" / "bd-icp"


def resolve_icp_dir(directory=None, config=None):
    if isinstance(directory, (str, Path)) and str(directory).strip():
        return Path(directory).expanduser()
    env = os.getenv(ENV_ICP_DIR, "")
    if isinstance(env, str) and env.strip():
        return Path(env).expanduser()
    if isinstance(config, dict):
        configured = config.get("icp_dir")
        if isinstance(configured, str) and configured.strip():
            return Path(configured).expanduser()
    return default_icp_dir()


def normalize_org_name(name):
    """Casefold, strip punctuation/suffixes/parentheticals, and unify &/and."""
    if not isinstance(name, str):
        return ""
    text = name.casefold().replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+[-–—]\s+.*$", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    tokens = [token for token in text.split() if token]
    if tokens and tokens[0] in {"the", "a", "an"}:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _contiguous_tokens(needles, haystack):
    if not needles or len(needles) > len(haystack):
        return False
    width = len(needles)
    for start in range(len(haystack) - width + 1):
        if haystack[start : start + width] == needles:
            return True
    return False


def org_names_match(candidate, suppressed):
    """Exact normalized match, else a conservative token-overlap/fuzzy match."""
    left = normalize_org_name(candidate)
    right = normalize_org_name(suppressed)
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens, right_tokens = left.split(), right.split()
    if len(left_tokens) >= 2 and len(right_tokens) >= 2:
        shorter, longer = (
            (left_tokens, right_tokens)
            if len(left_tokens) <= len(right_tokens)
            else (right_tokens, left_tokens)
        )
        if _contiguous_tokens(shorter, longer) and len(shorter) / len(longer) >= 0.7:
            return True
    shared = len(set(left_tokens) & set(right_tokens))
    return shared >= 2 and SequenceMatcher(None, left, right).ratio() >= 0.9


def matching_suppressed_org(name, organizations):
    if not isinstance(name, str) or not name.strip():
        return None
    for organization in organizations or ():
        if org_names_match(name, organization):
            return organization
    return None


def matches_suppression(name, organizations):
    return matching_suppressed_org(name, organizations) is not None


def _is_unit_ratio(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0 <= value <= 1


def sits_ratio(value):
    """Parse typical_ratio_of_sits: one number or a list of 1-2 numbers in [0, 1]."""
    if _is_unit_ratio(value):
        return (float(value),)
    if isinstance(value, list) and 1 <= len(value) <= 2:
        parsed = []
        for item in value:
            if isinstance(item, (list, dict, str, bool)) or not _is_unit_ratio(item):
                return None
            parsed.append(float(item))
        return tuple(parsed)
    return None


def sits_ratio_label(value):
    parsed = sits_ratio(value)
    if not parsed:
        return ""
    percents = [int(round(item * 100)) for item in parsed]
    low, high = min(percents), max(percents)
    if low == high:
        return f"{low}% of SITS"
    return f"{low}-{high}% of SITS"


def _strings(value, limit=8):
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
        if len(items) >= limit:
            break
    return items


def _quote_term(term):
    if " " in term or any(char in term for char in "()"):
        return '"' + term.replace('"', "") + '"'
    return term


def _or_group(terms):
    quoted = [_quote_term(term) for term in terms if term]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    return "(" + " OR ".join(quoted) + ")"


def query_terms(profile):
    """Compact Google News extras: hot buyers, sectors, priority geography."""
    icp = profile.icp if isinstance(profile, BotIcpProfile) else None
    if not icp:
        return ""
    buyers = icp.get("buyer_types") if isinstance(icp.get("buyer_types"), dict) else {}
    geography = icp.get("geography") if isinstance(icp.get("geography"), dict) else {}
    parts = [
        _or_group(_strings(buyers.get("hot"), 6)),
        _or_group(_strings(icp.get("employer_sectors"), 4)),
        _or_group(
            _strings(geography.get("priority_1"), 4) + _strings(geography.get("priority_2"), 3)
        ),
    ]
    extra = " ".join(part for part in parts if part)
    if len(extra) > 280:
        extra = extra[:280].rsplit(" ", 1)[0]
    return extra


@dataclass
class BotIcpProfile:
    bot_id: str
    loaded: bool = False
    status: str = "absent"
    error: str | None = None
    customer: str | None = None
    icp: dict | None = None
    organizations: tuple = field(default_factory=tuple)

    @property
    def suppression_count(self):
        return len(self.organizations)

    def summary(self, limit=1500):
        """Compact targeting guidance. No customer name and no suppression orgs."""
        if not self.loaded or not isinstance(self.icp, dict):
            return ""
        icp = self.icp
        buyers = icp.get("buyer_types") if isinstance(icp.get("buyer_types"), dict) else {}
        geography = icp.get("geography") if isinstance(icp.get("geography"), dict) else {}
        channels = icp.get("channels") if isinstance(icp.get("channels"), dict) else {}
        hot = ", ".join(_strings(buyers.get("hot")))
        down = ", ".join(_strings(buyers.get("downweight"), 6))
        priority = ", ".join(_strings(geography.get("priority_1"), 6))
        secondary = ", ".join(_strings(geography.get("priority_2"), 4))
        known = ", ".join(_strings(channels.get("known"), 6))
        treat = channels.get("treat_as")
        if not isinstance(treat, str) or not treat.strip():
            treat = "channel/partner, not end buyer"
        else:
            treat = treat.strip()
        product_bits = []
        products = icp.get("products")
        if isinstance(products, dict):
            for code, spec in list(products.items())[:6]:
                if not isinstance(code, str) or not code.strip():
                    continue
                if isinstance(spec, dict) and isinstance(spec.get("name"), str) and spec["name"]:
                    bit = f"{code.strip()}: {spec['name'].strip()}"
                    ratio = sits_ratio_label(spec.get("typical_ratio_of_sits"))
                    if ratio:
                        bit += f" ({ratio})"
                    product_bits.append(bit)
                else:
                    product_bits.append(code.strip())
        lines = ["ICP targeting guidance (not a prospect or customer list):"]
        if hot:
            lines.append(f"Prefer buyer types: {hot}.")
        if down:
            lines.append(f"Downweight lanes: {down}.")
        if priority or secondary:
            geo = f"Geography priority: {priority}" if priority else "Geography priority"
            if secondary:
                geo += f"; then {secondary}"
            lines.append(geo + ".")
        if known:
            lines.append(f"Channels ({known}) are {treat}.")
        else:
            lines.append(f"Brokers, co-ops, and health plans are {treat}.")
        if product_bits:
            lines.append("Product emphasis: " + "; ".join(product_bits) + ".")
        sectors = ", ".join(_strings(icp.get("employer_sectors"), 6))
        size = icp.get("employer_size") if isinstance(icp.get("employer_size"), str) else ""
        employer = []
        if sectors:
            employer.append(sectors)
        if size.strip():
            employer.append("size " + size.strip())
        if employer:
            lines.append("Employers: " + "; ".join(employer) + ".")
        text = " ".join(lines)
        if len(text) > limit:
            text = text[: limit - 3].rsplit(" ", 1)[0] + "..."
        return text

    def public_fields(self):
        """Snapshot projection: counts and customer label, never suppression names."""
        fields = {
            "icp_loaded": self.loaded,
            "icp_customer": self.customer if self.status != "error" else None,
            "icp_status": self.status,
            "suppression_count": 0 if self.status == "error" else self.suppression_count,
        }
        if self.status == "error" and self.error:
            fields["icp_error"] = self.error
        return fields


def activate(profile):
    usable = profile if isinstance(profile, BotIcpProfile) and profile.status == "ok" else None
    return _active_profile.set(usable)


def deactivate(token):
    _active_profile.reset(token)


def current_profile():
    return _active_profile.get()


def current_summary():
    profile = current_profile()
    return profile.summary() if profile and profile.loaded else ""


def current_organizations():
    profile = current_profile()
    return profile.organizations if profile else ()


def _valid_bot_id(bot_id):
    return isinstance(bot_id, str) and _BOT_ID.fullmatch(bot_id)


def _read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _nonempty_str(value):
    return isinstance(value, str) and bool(value.strip())


def _str_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _fail_closed_log(bot_id, error, key=None):
    if isinstance(key, str) and key:
        logger.warning("BD ICP for bot %s failed closed: %s key=%s", bot_id, error, key)
    else:
        logger.warning("BD ICP for bot %s failed closed: %s", bot_id, error)


def _validate_icp(data, bot_id):
    if data.get("version") != 1:
        return "malformed_icp_profile", "version"
    bot_ids = data.get("bot_ids")
    if (
        not isinstance(bot_ids, list)
        or not bot_ids
        or any(not _nonempty_str(item) for item in bot_ids)
    ):
        return "malformed_icp_profile", "bot_ids"
    if bot_id not in bot_ids:
        return "skip", None
    for key in ("customer", "updated", "source", "employer_size", "stages_note"):
        if not _nonempty_str(data.get(key)):
            return "malformed_icp_profile", key
    buyers = data.get("buyer_types")
    if (
        not isinstance(buyers, dict)
        or not _str_list(buyers.get("hot"))
        or not _str_list(buyers.get("downweight"))
    ):
        return "malformed_icp_profile", "buyer_types"
    if not _str_list(data.get("employer_sectors")):
        return "malformed_icp_profile", "employer_sectors"
    if not _str_list(data.get("signals")):
        return "malformed_icp_profile", "signals"
    geography = data.get("geography")
    if not isinstance(geography, dict) or any(
        not _str_list(geography.get(key)) for key in ("priority_1", "priority_2", "lower")
    ):
        return "malformed_icp_profile", "geography"
    products = data.get("products")
    if not isinstance(products, dict):
        return "malformed_icp_profile", "products"
    for spec in products.values():
        if not isinstance(spec, dict) or not _nonempty_str(spec.get("name")):
            return "malformed_icp_profile", "products"
        if "value_share" in spec:
            share = spec["value_share"]
            if (
                isinstance(share, bool)
                or not isinstance(share, (int, float))
                or not math.isfinite(share)
            ):
                return "malformed_icp_profile", "products"
        if "typical_ratio_of_sits" in spec and sits_ratio(spec["typical_ratio_of_sits"]) is None:
            return "malformed_icp_profile", "products"
    channels = data.get("channels")
    if (
        not isinstance(channels, dict)
        or not _nonempty_str(channels.get("treat_as"))
        or not _str_list(channels.get("known"))
    ):
        return "malformed_icp_profile", "channels"
    return "ok", None


def _validate_suppression(data, bot_id):
    if data.get("version") != 1:
        return "malformed_suppression_list", "version"
    bot_ids = data.get("bot_ids")
    if (
        not isinstance(bot_ids, list)
        or not bot_ids
        or any(not _nonempty_str(item) for item in bot_ids)
    ):
        return "malformed_suppression_list", "bot_ids"
    if bot_id not in bot_ids:
        return "skip", None
    for key in ("customer", "reason", "updated"):
        if not _nonempty_str(data.get(key)):
            return "malformed_suppression_list", key
    if not _str_list(data.get("organizations")):
        return "malformed_suppression_list", "organizations"
    if "notes" in data and not isinstance(data["notes"], str):
        return "malformed_suppression_list", "notes"
    return "ok", None


def load_bot_profile(bot_id, *, directory=None, config=None):
    """Load matching ICP/suppression files. Never raises; never logs file bodies."""
    label = bot_id if isinstance(bot_id, str) else ""
    absent = BotIcpProfile(bot_id=label, status="absent")
    if not _valid_bot_id(bot_id):
        return absent
    try:
        root = resolve_icp_dir(directory, config)
        if not root.exists():
            return absent
        if not root.is_dir():
            _fail_closed_log(bot_id, "unreadable_icp_dir")
            return BotIcpProfile(bot_id=bot_id, status="error", error="unreadable_icp_dir")
        folder = root / bot_id
        if not folder.is_dir():
            return absent
        icp_files = sorted(path for path in folder.glob("*.icp.json") if path.is_file())
        suppression_files = sorted(
            path for path in folder.glob("*.suppression.json") if path.is_file()
        )
        if not icp_files and not suppression_files:
            return absent
        icp_data = None
        customer = None
        organizations = []
        seen = set()
        for path in icp_files:
            parsed = _read_json(path)
            if parsed is None:
                _fail_closed_log(bot_id, "malformed_icp_profile")
                return BotIcpProfile(bot_id=bot_id, status="error", error="malformed_icp_profile")
            check, key = _validate_icp(parsed, bot_id)
            if check == "skip":
                continue
            if check != "ok":
                _fail_closed_log(bot_id, check, key)
                return BotIcpProfile(bot_id=bot_id, status="error", error=check)
            icp_data = parsed
            customer = parsed["customer"].strip()
            break
        for path in suppression_files:
            parsed = _read_json(path)
            if parsed is None:
                _fail_closed_log(bot_id, "malformed_suppression_list")
                return BotIcpProfile(
                    bot_id=bot_id, status="error", error="malformed_suppression_list"
                )
            check, key = _validate_suppression(parsed, bot_id)
            if check == "skip":
                continue
            if check != "ok":
                _fail_closed_log(bot_id, check, key)
                return BotIcpProfile(bot_id=bot_id, status="error", error=check)
            if customer is None:
                customer = parsed["customer"].strip()
            for organization in parsed["organizations"]:
                if not organization.strip() or organization in seen:
                    continue
                seen.add(organization)
                organizations.append(organization)
                if len(organizations) >= 2000:
                    break
        if icp_data is None and not organizations and customer is None:
            return absent
        return BotIcpProfile(
            bot_id=bot_id,
            loaded=icp_data is not None,
            status="ok",
            customer=customer,
            icp=icp_data,
            organizations=tuple(organizations),
        )
    except (OSError, TypeError, ValueError):
        _fail_closed_log(bot_id, "malformed_icp_profile")
        return BotIcpProfile(bot_id=bot_id, status="error", error="malformed_icp_profile")
