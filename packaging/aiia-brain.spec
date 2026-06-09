# PyInstaller spec for the standalone AIIA Brain binary (`aiia-brain`).
#
# Build on the target Mac (binaries are per-arch, no cross-compile):
#   pip install -e ".[dev]" pyinstaller
#   pyinstaller packaging/aiia-brain.spec
#
# Output: dist/aiia-brain — a single-file binary that aiia-console ships as a
# Tauri sidecar (see aiia-console docs/PACKAGING.md). The console expects the
# Tauri externalBin naming convention, e.g. aiia-brain-aarch64-apple-darwin;
# packaging/build-brain.sh handles the rename.
#
# ChromaDB + tiktoken are the PyInstaller-hostile deps: both load data files
# and plugins dynamically, so we collect them wholesale rather than chasing
# individual hidden imports as versions move.

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    # uvicorn's import-string workers
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    # tiktoken registers encodings via a namespace package
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
]

for pkg in ("chromadb", "onnxruntime", "tokenizers", "local_brain"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["../local_brain/standalone.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="aiia-brain",
    console=True,
    upx=False,
    target_arch=None,  # native arch of the build machine
)
