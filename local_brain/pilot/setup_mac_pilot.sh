#!/bin/bash
# ─────────────────────────────────────────────────────────────
# AIIA — Mac Mini Pilot Setup
#
# Run this on your Mac Mini to set up AIIA:
#   1. Ollama (local LLM runtime)
#   2. Models (qwen2.5:7b, nomic-embed-text)
#   3. AIIA Local Brain API (FastAPI service)
#   4. AIIA Knowledge + Memory (ChromaDB + JSON)
#   5. Tailscale (secure tunnel to production)
#   6. launchd auto-start (survives reboots)
#
# Usage:
#   chmod +x setup_mac_pilot.sh
#   ./setup_mac_pilot.sh
# ─────────────────────────────────────────────────────────────

set -e

echo "╔═══════════════════════════════════════════════════╗"
echo "║  AIIA — Mac Mini Pilot Setup                      ║"
echo "║  AI Information Architecture + Ollama              ║"
echo "║  The persistent AI teammate for your platform      ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

step() { echo -e "\n${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

# ─────────────────────────────────────────────────────────────
# Prerequisites
# ─────────────────────────────────────────────────────────────

echo "Checking prerequisites..."

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    fail "This script is for macOS only."
fi

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    warn "Not running on Apple Silicon. Performance will be limited."
fi

# Check memory
TOTAL_MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
step "Detected ${TOTAL_MEM_GB}GB unified memory"
if [[ $TOTAL_MEM_GB -lt 16 ]]; then
    warn "16GB+ recommended for local LLMs. You have ${TOTAL_MEM_GB}GB."
fi

# ─────────────────────────────────────────────────────────────
# Step 1: Homebrew (if needed)
# ─────────────────────────────────────────────────────────────

if ! command -v brew &> /dev/null; then
    step "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    step "Homebrew already installed"
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Ollama
# ─────────────────────────────────────────────────────────────

if ! command -v ollama &> /dev/null; then
    step "Installing Ollama..."
    brew install ollama
else
    step "Ollama already installed"
fi

# Start Ollama service
step "Starting Ollama service..."
brew services start ollama 2>/dev/null || true
sleep 3  # Give it a moment to start

# Verify Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    step "Ollama is running on :11434"
else
    warn "Ollama not responding yet. It may need a moment..."
    sleep 5
fi

# ─────────────────────────────────────────────────────────────
# Step 3: Pull Models
# ─────────────────────────────────────────────────────────────

step "Pulling models (this takes a few minutes on first run)..."

echo "  Pulling qwen2.5:7b (routing + tasks — best structured output at 7B)..."
ollama pull qwen2.5:7b

echo "  Pulling nomic-embed-text (embeddings for EQ Brain knowledge store)..."
ollama pull nomic-embed-text

step "Models ready:"
ollama list

# ─────────────────────────────────────────────────────────────
# Step 4: Python Environment for Local Brain
# ─────────────────────────────────────────────────────────────

step "Setting up Python environment for AIIA..."

# Ensure Python 3.11+
if ! command -v python3 &> /dev/null; then
    brew install python@3.12
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
step "Python ${PYTHON_VERSION}"

# Create virtual environment
BRAIN_DIR="$HOME/.aiia"
mkdir -p "$BRAIN_DIR"

if [ ! -d "$BRAIN_DIR/venv" ]; then
    python3 -m venv "$BRAIN_DIR/venv"
    step "Created virtual environment at $BRAIN_DIR/venv"
else
    step "Virtual environment already exists"
fi

source "$BRAIN_DIR/venv/bin/activate"

# Install dependencies (includes ChromaDB for EQ Brain knowledge store)
pip install --quiet --upgrade pip
pip install --quiet \
    fastapi uvicorn httpx pydantic \
    chromadb \
    pyyaml \
    aiofiles

step "Python dependencies installed (AIIA stack: FastAPI + ChromaDB + utilities)"

# ─────────────────────────────────────────────────────────────
# Step 5: Clone/Update Codebase
# ─────────────────────────────────────────────────────────────

REPO_DIR="$BRAIN_DIR/AIIA"
if [ ! -d "$REPO_DIR" ]; then
    step "Cloning AIIA repository..."
    git clone https://github.com/OWNER/AIIA.git "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout main
else
    step "Updating existing repository..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout main
    git pull origin main
fi

step "Codebase ready at $REPO_DIR"

# ─────────────────────────────────────────────────────────────
# Step 6: Environment Configuration
# ─────────────────────────────────────────────────────────────

ENV_FILE="$BRAIN_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    step "Creating environment configuration..."
    cat > "$ENV_FILE" << 'ENVEOF'
# ─────────────────────────────────────────────────────────────
# AIIA Local Brain — Environment Configuration
# ─────────────────────────────────────────────────────────────

# Ollama (local LLM runtime)
LOCAL_LLM_URL=http://localhost:11434

# Local Brain API
LOCAL_BRAIN_HOST=0.0.0.0
LOCAL_BRAIN_PORT=8100
# LOCAL_BRAIN_API_KEY=your-secret-key-here  # Uncomment for production

# Model assignments (qwen2.5:7b — best structured output at 7B class)
LOCAL_ROUTING_MODEL=qwen2.5:7b
LOCAL_TASK_MODEL=qwen2.5:7b
LOCAL_EMBED_MODEL=nomic-embed-text

# AIIA — persistent AI teammate (knowledge + memory)
EQ_BRAIN_ENABLED=true
EQ_BRAIN_DATA_DIR=$HOME/.aiia/eq_data
EQ_BRAIN_COLLECTION=aiia_knowledge

# Feature flags
LOCAL_SMART_ROUTING=true
LOCAL_SUMMARIZATION=true
LOCAL_MEMORY_EXTRACTION=true
LOCAL_PII_SCANNING=true
LOCAL_EMBEDDINGS=true
ENVEOF
    step "Environment file created at $ENV_FILE"
    warn "Edit $ENV_FILE to set API key for production use"
else
    step "Environment file already exists"
fi

# Create AIIA data directory
EQ_DATA_DIR="$BRAIN_DIR/eq_data"
mkdir -p "$EQ_DATA_DIR/chroma"
mkdir -p "$EQ_DATA_DIR/memory"
step "AIIA data directory created at $EQ_DATA_DIR"

# ─────────────────────────────────────────────────────────────
# Step 7: Tailscale (secure tunnel)
# ─────────────────────────────────────────────────────────────

if ! command -v tailscale &> /dev/null; then
    step "Installing Tailscale..."
    brew install --cask tailscale
    warn "Open Tailscale app and sign in to connect to your network"
else
    step "Tailscale already installed"
fi

# ─────────────────────────────────────────────────────────────
# Step 8: Bootstrap EQ Brain Knowledge
# ─────────────────────────────────────────────────────────────

step "Bootstrapping AIIA with repo knowledge..."
info "This indexes CLAUDE.md, architecture docs, and codebase structure"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export EQ_BRAIN_DATA_DIR="$EQ_DATA_DIR"

python3 -c "
import asyncio
from local_brain.eq_brain.bootstrap import bootstrap_from_repo
asyncio.run(bootstrap_from_repo('$REPO_DIR'))
print('AIIA knowledge bootstrapped successfully.')
" 2>/dev/null || warn "AIIA bootstrap will run on first start (module may need repo update)"

# ─────────────────────────────────────────────────────────────
# Step 9: Create launch script
# ─────────────────────────────────────────────────────────────

LAUNCH_SCRIPT="$BRAIN_DIR/start_brain.sh"
cat > "$LAUNCH_SCRIPT" << 'LAUNCHEOF'
#!/bin/bash
# Start AIIA — the persistent AI teammate for your platform.
# Run this to bring the full stack online.

set -e

BRAIN_DIR="$HOME/.aiia"
source "$BRAIN_DIR/.env"
source "$BRAIN_DIR/venv/bin/activate"

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║       A I I A — Waking Up...                      ║"
echo "║  AI Information Architecture                       ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Ensure Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    brew services start ollama
    sleep 3
fi

# Show loaded models
echo "Ollama models:"
ollama list
echo ""

# Start AIIA Local Brain API
cd "$BRAIN_DIR/AIIA"
export PYTHONPATH="$BRAIN_DIR/AIIA:$PYTHONPATH"
export EQ_BRAIN_DATA_DIR="$BRAIN_DIR/eq_data"

echo "Starting AIIA on :8100..."
python -m uvicorn local_brain.local_api:app \
    --host ${LOCAL_BRAIN_HOST:-0.0.0.0} \
    --port ${LOCAL_BRAIN_PORT:-8100} &
BRAIN_PID=$!

sleep 2
echo "Starting Command Center on :8200..."
python -m uvicorn local_brain.command_center.server:app \
    --host 0.0.0.0 \
    --port 8200 &
CMD_PID=$!

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║       A I I A — Online                            ║"
echo "║  AI Information Architecture                       ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║  Ollama:          http://localhost:11434           ║"
echo "║  AIIA API:        http://localhost:8100            ║"
echo "║  Command Center:  http://localhost:8200            ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║  AIIA endpoints:                                  ║"
echo "║    POST /v1/aiia/ask      — Ask AIIA anything     ║"
echo "║    POST /v1/aiia/remember — Teach her something   ║"
echo "║    GET  /v1/aiia/status   — Knowledge + memory    ║"
echo "║    POST /v1/aiia/ingest   — Index new documents   ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "  Dashboard: http://localhost:8200"
echo "  Press Ctrl+C to stop all services."
echo ""

# Trap Ctrl+C to kill both services
trap "kill $BRAIN_PID $CMD_PID 2>/dev/null; exit" INT TERM
wait
LAUNCHEOF
chmod +x "$LAUNCH_SCRIPT"
step "Launch script created at $LAUNCH_SCRIPT"

# ─────────────────────────────────────────────────────────────
# Step 10: launchd auto-start (survives reboots)
# ─────────────────────────────────────────────────────────────

step "Setting up auto-start on boot..."

PLIST_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$PLIST_DIR"

PLIST_FILE="$PLIST_DIR/com.aiia.localbrain.plist"
cat > "$PLIST_FILE" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aiia.localbrain</string>
    <key>ProgramArguments</key>
    <array>
        <string>${BRAIN_DIR}/start_brain.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${BRAIN_DIR}/logs/brain.log</string>
    <key>StandardErrorPath</key>
    <string>${BRAIN_DIR}/logs/brain_error.log</string>
    <key>WorkingDirectory</key>
    <string>${BRAIN_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>${HOME}</string>
    </dict>
</dict>
</plist>
PLISTEOF

mkdir -p "$BRAIN_DIR/logs"

# Load the launch agent
launchctl unload "$PLIST_FILE" 2>/dev/null || true
launchctl load "$PLIST_FILE"
step "Auto-start configured — brain will start on every boot"
info "Logs at: $BRAIN_DIR/logs/brain.log"
info "To disable: launchctl unload $PLIST_FILE"

# ─────────────────────────────────────────────────────────────
# Step 11: Quick test
# ─────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────"
step "Running quick Ollama test..."
echo ""

# Test with a simple prompt
RESPONSE=$(curl -s http://localhost:11434/api/chat \
    -d '{
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
        "stream": false
    }' 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',{}).get('content','No response'))" 2>/dev/null || echo "Ollama not ready yet")

echo "  Ollama says: $RESPONSE"
echo ""

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────

echo "╔═══════════════════════════════════════════════════╗"
echo "║       A I I A — Setup Complete                    ║"
echo "║  The persistent AI teammate is ready.              ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "  Workspace:     $BRAIN_DIR"
echo "  Codebase:      $REPO_DIR"
echo "  AIIA Data:     $EQ_DATA_DIR"
echo "  Environment:   $ENV_FILE"
echo "  Logs:          $BRAIN_DIR/logs/"
echo ""
echo "  Services (auto-starting on boot):"
echo "    Ollama:          http://localhost:11434"
echo "    AIIA API:        http://localhost:8100"
echo "    Command Center:  http://localhost:8200"
echo ""
echo "  Quick commands:"
echo "    curl http://localhost:8100/health            # Health check"
echo "    curl http://localhost:8100/v1/aiia/status    # AIIA stats"
echo "    tail -f $BRAIN_DIR/logs/brain.log            # Watch logs"
echo ""
echo "  Then on your production server, set:"
echo "    LOCAL_LLM_URL=http://<your-tailscale-ip>:11434"
echo "    LOCAL_BRAIN_URL=http://<your-tailscale-ip>:8100"
echo ""
echo "  AIIA is awake. Let's build."
echo ""
