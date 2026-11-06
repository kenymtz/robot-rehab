#!/usr/bin/env bash
# Usage: ./run.sh [--smoke] [--foreground]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

UV_VERSION=0.12.16
CONFIG=configs/full.yaml
FOREGROUND=0
for arg in "$@"; do
    case "$arg" in
        --smoke) CONFIG=configs/smoke.yaml; FOREGROUND=1 ;;
        --foreground) FOREGROUND=1 ;;
        *) echo "Opción desconocida: $arg" >&2; exit 2 ;;
    esac
done

if [ "$FOREGROUND" -eq 0 ]; then
    setsid nohup bash run.sh --foreground >> run.log 2>&1 < /dev/null &
    echo "Corriendo en segundo plano (PID $!). Ver progreso con: tail -f run.log"
    exit 0
fi

export UV_UNMANAGED_INSTALL="$PWD/.tools/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_CACHE_DIR="$PWD/.tools/cache"
export UV_PYTHON_PREFERENCE=only-managed

if [ ! -x .tools/uv/uv ]; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
fi
.tools/uv/uv sync --frozen

COMMAND=(.tools/uv/uv run --frozen python -m robotrehab.cli --config "$CONFIG")
if command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --what=sleep:idle true 2>/dev/null; then
    exec systemd-inhibit --what=sleep:idle --who=robot-rehab --why="corrida larga" "${COMMAND[@]}"
fi
exec "${COMMAND[@]}"
