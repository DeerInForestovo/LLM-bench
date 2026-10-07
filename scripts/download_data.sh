#!/usr/bin/env bash
# Data acquisition script for LLM-bench.
#
# Downloads the official logo of each evaluated model into data/raw/.
# Logos are fetched from each vendor's public web presence (favicon /
# PWA icon endpoints), which guarantees that we evaluate the exact visual
# identity that the vendor ships to users.
#
# Usage: bash scripts/download_data.sh
set -euo pipefail

RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"

fetch_favicon() {
    # $1: output name, $2: domain
    echo "[download] $1 <- $2"
    curl -sL --max-time 30 -o "$RAW_DIR/$1.png" \
        "https://www.google.com/s2/favicons?domain=$2&sz=256"
}

fetch_favicon openai    openai.com
fetch_favicon claude    claude.ai
fetch_favicon gemini    gemini.google.com
fetch_favicon grok      grok.com
fetch_favicon mistral   mistral.ai
fetch_favicon llama     llama.com
fetch_favicon deepseek  deepseek.com
fetch_favicon doubao    doubao.com
fetch_favicon qwen      qwen.ai

# Kimi's favicon endpoint serves a JPEG with heavy compression artifacts;
# the official PWA icon distributed on kimi.com is a lossless PNG.
echo "[download] kimi <- kimi.com (pwa-192.png)"
curl -sL --max-time 30 -A "Mozilla/5.0" \
    -o "$RAW_DIR/kimi.png" "https://www.kimi.com/pwa-192.png"

echo "[download] done. $(ls "$RAW_DIR" | wc -l) files in $RAW_DIR"
