# AIIA Journal — voice-first journaling, anywhere

AIIA Journal is the part of AIIA that turns a voice memo into a structured Markdown entry in your Obsidian vault. It runs three ways, all writing to the same vault:

| Surface | Where it runs | Best for |
|---|---|---|
| **aiia-console** (desktop) | Mac mini (or any macOS dev machine) | At-the-desk sessions, full Memory + Settings access |
| **iCloud + iOS Shortcut** | Phone → iCloud → mini's watcher daemon | On-the-go capture (walking, in transit, anywhere) |
| **`POST /v1/journal/ingest`** | Any device that can reach the mini | Future PWA, native iOS app, automations |

This doc explains the mobile path — the one you set up once and use forever.

## Setup on the mini

### 1. API keys

The pipeline needs two keys: one for Whisper (transcription) and one for distillation.

```bash
# Required — for Whisper STT
launchctl setenv GROQ_API_KEY 'gsk_...'

# Recommended — Claude writes journal-shaped prose better than the others
launchctl setenv ANTHROPIC_API_KEY 'sk-ant-...'

# Optional fallbacks if you don't want Anthropic
# launchctl setenv OPENAI_API_KEY 'sk-...'
# (Groq is already set — it's the same key used for Whisper)
```

`launchctl setenv` persists across logins, so you do this once.

### 2. Install the watcher

```bash
cd ~/code/aiia
bash scripts/install-journal-watcher.sh
```

That copies a LaunchAgent into `~/Library/LaunchAgents/`, loads it, and starts polling. It will:

- Watch `~/Library/Mobile Documents/com~apple~CloudDocs/AIIA-Inbox/` every 5s.
- Pick up any new audio file (m4a, mp3, wav, webm, ogg, flac).
- Run the transcribe → distill → vault-write pipeline.
- Move the source audio into `AIIA-Inbox/archived/`.

You can verify it's running:

```bash
launchctl print "gui/$(id -u)/com.aplora.aiia.journal-watch"
tail -f /tmp/aiia-journal-watch.err
```

### 3. Make sure iCloud Drive is enabled

The watcher folder lives in iCloud Drive, so this needs to be on for both the mini and the phone. macOS: System Settings → Apple ID → iCloud → iCloud Drive ON. iOS: Settings → [your name] → iCloud → iCloud Drive ON.

## Setup on the phone — the iOS Shortcut

This Shortcut is the entire "AIIA Journal mobile app." Add it to your home screen and you're done.

**To create it:**

1. Open the **Shortcuts** app on your iPhone.
2. Tap **+** (new shortcut).
3. Add these actions in order:

   | # | Action | Configuration |
   |---|---|---|
   | 1 | **Record Audio** | Audio Quality: *Normal* (m4a). Stop Recording: *On Tap*. |
   | 2 | **Get File Name** | (input: Recorded Audio) |
   | 3 | **Text** | `journal-` then **Format Date** (Current Date, ISO 8601 with time, custom format `yyyy-MM-dd-HHmmss`) then `.m4a` |
   | 4 | **Save File** | Input: *Recorded Audio*. Service: *iCloud Drive*. Destination Path: `AIIA-Inbox/[text from step 3]`. Ask Where to Save: *off*. Overwrite If File Exists: *off*. |
   | 5 | (optional) **Show Notification** | "Journal sent — it'll show up in your vault in a minute." |

4. Tap the settings icon → name it **"AIIA Journal"** → choose a glyph (mic icon recommended).
5. Tap **Add to Home Screen** so it lives one tap away.

**To use it:**

Tap the home-screen icon. Talk into the phone. Tap "Stop Recording" when done. iCloud takes ~10-60 seconds to upload (faster on Wi-Fi). The mini's watcher picks it up, runs the pipeline, and the markdown lands in `00-Inbox/YYYY-MM-DD-HHMMSS-session.md` in your vault, usually within 2 minutes of recording.

**Tips:**

- The Shortcut works offline too. Recordings queue in iCloud and upload when you reconnect.
- If you want to journal during a walk and skip the home-screen tap, ask Siri: "AIIA Journal" (the Shortcut name doubles as the Siri phrase).
- The watcher is idempotent on filenames — the timestamp prefix means two recordings can't collide.

## Setup on the phone — the HTTP path (advanced)

If your mini is reachable from your phone (Tailscale, public IP behind a tunnel, etc.), you can skip iCloud and POST audio directly:

```bash
curl -X POST "http://[mini-tailscale-ip]:8100/v1/journal/ingest" \
  -F "audio=@recording.m4a;type=audio/mp4" \
  -F "duration_seconds=120" \
  -F "started_at=2026-05-22T21:30:00-07:00"
```

Response shape:

```json
{
  "vault_path": "/Users/.../Documents/AIIA/00-Inbox/2026-05-22-213000-session.md",
  "vault_relative": "00-Inbox/2026-05-22-213000-session.md",
  "distilled": true,
  "transcription": { "provider": "groq", "model": "whisper-large-v3-turbo", "text": "..." },
  "distillation": { "provider": "anthropic", "model": "claude-sonnet-4-5" },
  "markdown": "---\n..."
}
```

A future iOS Shortcut, PWA, or native app can use this endpoint instead of the iCloud-folder path. The pipeline is identical either way.

## Output format

Every journal session lands as a markdown file with this frontmatter:

```yaml
---
date: 2026-05-22
started: 2026-05-22T21:30:00-07:00
duration_seconds: 423
transcription: groq
aiia_managed: true
tags: [journal, session]
---
```

The body has four optional sections (the distiller skips any that don't apply):

- **Title + opening** — 4-7 word title, 2-3 sentence scene-setting
- **What I'm working through** — 2-4 paragraphs of prose, first person
- **Threads** — bullets of topics the session touched
- **Decisions** — only if decisions were actually made
- **Open questions** — what's still unresolved

If distillation fails for any reason (LLM provider down, network glitch), the file is still written with the raw transcript and `tags: [journal, session, raw]`. The session is never lost.

## Troubleshooting

**Audio shows up in `AIIA-Inbox/` but doesn't get processed.** Check `tail /tmp/aiia-journal-watch.err`. Most common: `GROQ_API_KEY` not set in the LaunchAgent's environment. Re-run `launchctl setenv GROQ_API_KEY '...'` and kick the watcher: `launchctl kickstart -k gui/$(id -u)/com.aplora.aiia.journal-watch`.

**Recordings never reach the mini.** iCloud Drive sync issue. Check that both devices show the same files in the Files app (iOS) and Finder (macOS). If iCloud is paused (low storage, "Optimize Mac Storage" off), recordings will queue indefinitely.

**Watcher restarts in a loop.** Likely a Python import error. Check `tail /tmp/aiia-journal-watch.err` for a traceback. Most common: `aiia` not on the PATH that `bash -lc` uses inside the LaunchAgent. Adjust the plist's `EnvironmentVariables → PATH` to include your venv's `bin/`.

**Transcripts are missing words / wrong.** Whisper accuracy depends on input quality. Iterate: speak closer to the mic, reduce background noise. The `?language=en` hint in the Shortcut (add a Text action and pass it as a multipart field) bumps accuracy on short clips.

**The distilled markdown feels too formal.** Adjust `local_brain/journal/distiller.py::_system_prompt`. The current prompt errs on the side of structure; if you want more conversational, drop the bullet-list sections and rewrite the structural guidance.

## Related

- The desktop console's journaling surface lives in `ericlovo/aiia-console` (`src/components/JournalTab.tsx`). It uses the same Groq Whisper API and the same distillation prompt shape, but writes to the vault directly via the Tauri keystore + vault_write commands rather than going through this Python pipeline. Both produce structurally identical markdown.
- `local_brain.vault_paths.vault_dir()` is the canonical vault root resolution used by everything that writes to your Obsidian vault. Honors `OBSIDIAN_VAULT_DIR` first, then `~/Documents/AIIA`, then `~/.aiia/vault`.
