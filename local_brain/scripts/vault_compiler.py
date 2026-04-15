"""
Vault Compiler — Ingest raw sources from 00-Inbox/ into the AIIA knowledge wiki.

Reads files from the vault's 00-Inbox/ folder, uses the local LLM to
summarize and extract key facts, writes wiki articles to 85-Wiki/,
stores extracted facts as AIIA memories, and moves processed files to
00-Inbox/processed/.

Usage:
    python -m local_brain.scripts.vault_compiler
    python -m local_brain.scripts.vault_compiler --dry-run

Supports: .md, .txt files directly. .pdf and .docx via macOS textutil.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# --- Vault path resolution ---

from local_brain.vault_paths import vault_dir as _vault_dir

VAULT_DIR = _vault_dir()
INBOX_DIR = VAULT_DIR / "00-Inbox"
PROCESSED_DIR = INBOX_DIR / "processed"
WIKI_DIR = VAULT_DIR / "85-Wiki"
BRAIN_URL = os.getenv("AIIA_URL", "http://localhost:8100")

# Max chars of source text to send to LLM
MAX_SOURCE_CHARS = 20_000


def _slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return slug.strip("-")[:max_len].rstrip("-") or "untitled"


def _extract_text(path: Path) -> Optional[str]:
    """Extract text content from a file. Returns None if unsupported."""
    suffix = path.suffix.lower()

    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_SOURCE_CHARS]

    if suffix in (".pdf", ".docx", ".doc", ".rtf"):
        # Use macOS textutil for conversion
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["textutil", "-convert", "txt", "-output", tmp_path, str(path)],
                capture_output=True,
                timeout=30,
            )
            text = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
            os.unlink(tmp_path)
            return text[:MAX_SOURCE_CHARS]
        except Exception as e:
            print(f"  textutil failed for {path.name}: {e}")
            return None

    return None


def _call_ollama(prompt: str, system: str = "", max_tokens: int = 4096) -> Optional[str]:
    """Call local Ollama via HTTP (no httpx dependency)."""
    url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434") + "/api/generate"
    payload = json.dumps(
        {
            "model": os.getenv("LOCAL_TASK_MODEL", "llama3.1:8b-instruct-q8_0"),
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": max_tokens,
                "num_ctx": 32768,
            },
        }
    ).encode()

    try:
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "")
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None


def _remember_fact(fact: str, category: str = "lessons", source: str = "vault-compiler"):
    """Store a fact in AIIA memory via the Brain API."""
    url = f"{BRAIN_URL}/v1/aiia/remember"
    payload = json.dumps(
        {
            "fact": fact,
            "category": category,
            "source": source,
        }
    ).encode()
    try:
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception:
        pass  # Best-effort


def _parse_llm_response(response: str) -> Dict:
    """Parse the LLM's structured JSON response."""
    # Try to extract JSON block
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Fallback: extract what we can
    return {"title": "", "summary": response[:2000], "key_facts": [], "folder": "85-Wiki"}


COMPILE_PROMPT = """Analyze this document and produce a structured summary for a knowledge wiki.

Return a JSON object with these fields:
```json
{{
  "title": "A concise, descriptive title for the wiki article",
  "summary": "A comprehensive wiki-style summary (500-2000 words). Include key concepts, decisions, and actionable information. Use markdown formatting.",
  "key_facts": [
    "Fact 1 — a standalone piece of knowledge worth remembering",
    "Fact 2 — another key insight or decision"
  ],
  "category": "decisions|patterns|lessons|project",
  "folder": "85-Wiki",
  "suggested_links": ["existing-page-name-1", "existing-page-name-2"]
}}
```

Key facts should be self-contained sentences that make sense without the original document.
The category should match: decisions (for choices/architecture), patterns (for conventions/approaches),
lessons (for hard-won insights), or project (for status/milestones).

DOCUMENT:
{text}"""


def compile_file(path: Path, dry_run: bool = False) -> bool:
    """Process a single file from the inbox. Returns True on success."""
    print(f"\nProcessing: {path.name}")

    # Extract text
    text = _extract_text(path)
    if not text or len(text.strip()) < 50:
        print(f"  Skipped: insufficient text content ({len(text or '')} chars)")
        return False

    print(f"  Extracted {len(text)} chars")

    if dry_run:
        print(f"  [dry-run] Would compile with LLM and write wiki article")
        return True

    # LLM compilation
    prompt = COMPILE_PROMPT.format(text=text[:MAX_SOURCE_CHARS])
    response = _call_ollama(
        prompt,
        system="You are a knowledge compiler. Extract and organize information into structured wiki articles. Return valid JSON only.",
    )
    if not response:
        print(f"  Failed: no LLM response")
        return False

    parsed = _parse_llm_response(response)
    title = parsed.get("title") or path.stem.replace("-", " ").replace("_", " ").title()
    summary = parsed.get("summary", "")
    key_facts = parsed.get("key_facts", [])
    category = parsed.get("category", "lessons")
    suggested_links = parsed.get("suggested_links", [])

    if not summary:
        print(f"  Failed: empty summary from LLM")
        return False

    print(f"  Title: {title}")
    print(f"  Summary: {len(summary)} chars, {len(key_facts)} facts extracted")

    # Write wiki article
    today = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(title)
    filename = f"{slug}.md"
    dest = WIKI_DIR / filename

    # Build wikilinks
    links_section = ""
    if suggested_links:
        link_lines = [f"- [[{l}]]" for l in suggested_links[:5]]
        links_section = "\n## Related\n" + "\n".join(link_lines) + "\n"

    content = "\n".join(
        [
            "---",
            "type: aiia-wiki",
            f"date: {today}",
            f"source: vault-compiler",
            "aiia_managed: true",
            "aiia_version: 2",
            f'compiled_from: "{path.name}"',
            f"tags: [aiia, wiki, {category}]",
            "---",
            "",
            f"# {title}",
            "",
            "> Compiled from `{path.name}` by AIIA Vault Compiler.",
            "",
            summary,
            "",
            links_section,
            f"*Compiled {today}*",
        ]
    )

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.rename(str(tmp), str(dest))
    print(f"  Written: 85-Wiki/{filename} ({len(content)} chars)")

    # Store key facts as AIIA memories
    for fact in key_facts[:10]:
        if len(fact) > 20:
            _remember_fact(fact, category=category, source=f"vault-compiler:{path.name}")
            print(f"  Remembered: {fact[:80]}...")

    # Move to processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest_processed = PROCESSED_DIR / path.name
    if dest_processed.exists():
        dest_processed = PROCESSED_DIR / f"{path.stem}_{today}{path.suffix}"
    shutil.move(str(path), str(dest_processed))
    print(f"  Moved to: 00-Inbox/processed/{dest_processed.name}")

    return True


def run(dry_run: bool = False) -> Dict:
    """Process all files in 00-Inbox/."""
    print(f"Vault Compiler — inbox={INBOX_DIR} dry_run={dry_run}")

    if not INBOX_DIR.exists():
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created inbox directory: {INBOX_DIR}")
        print("Drop .md, .txt, .pdf, or .docx files here, then run `brain compile`.")
        return {"processed": 0, "failed": 0, "skipped": 0}

    # Find files to process (skip processed/ subdirectory)
    files = [
        f
        for f in sorted(INBOX_DIR.iterdir())
        if f.is_file()
        and not f.name.startswith(".")
        and f.suffix.lower() in (".md", ".txt", ".pdf", ".docx", ".doc", ".rtf")
    ]

    if not files:
        print("No files to process in 00-Inbox/")
        return {"processed": 0, "failed": 0, "skipped": 0}

    print(f"Found {len(files)} files to compile")

    processed = 0
    failed = 0
    for f in files:
        try:
            if compile_file(f, dry_run=dry_run):
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")
            failed += 1

    print(f"\nDone: {processed} compiled, {failed} failed")
    return {"processed": processed, "failed": failed}


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    run(dry_run=dry_run)


if __name__ == "__main__":
    main()
