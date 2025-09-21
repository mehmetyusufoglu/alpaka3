#!/usr/bin/env bash
set -euo pipefail

# Generate all PlantUML diagrams in diagrams/ into output/
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DIAGRAMS_DIR="$ROOT_DIR/diagrams"
OUTPUT_DIR="$ROOT_DIR/output"

mkdir -p "$OUTPUT_DIR"

if ! command -v plantuml >/dev/null 2>&1; then
  echo "plantuml is required. Install it and ensure it's on PATH." >&2
  exit 1
fi

find "$DIAGRAMS_DIR" -maxdepth 1 -type f -name '*.puml' | while read -r f; do
  echo "Generating $(basename "$f")"
  plantuml -tpng -o "$OUTPUT_DIR" "$f"
  plantuml -tsvg -o "$OUTPUT_DIR" "$f"
done
